import os
import numpy as np
import torch
import nvdiffrast.torch as dr
from . import util
from . import renderutils as ru
from utils.general_utils import get_homogeneous
from utils.sh_utils import  eval_sh
from utils.sh_additional_utils import sh_render
from utils.sh_utils import gauss_kernel
from typing import Dict, Tuple


class EnvironmentLight(torch.nn.Module):

    def __init__(self, base: torch.Tensor, sh_degree : int = 4):
        """
        The class implements a Cook-Torrance shader based on IBL by following the implementation of NVDIFFREC, https://github.com/NVlabs/nvdiffrecmc.

        Attributes:
            base (torch.tesnor): Spherical Harmonics (SH) coefficients
            sh_degree (int): SH degree,
            sh_dim (int): number of SH coefficients.
        Constants
            NUM_CHANNELS (int): number of channels of base, which is RGB,
            C1,C2,...,C5 (int): constants for computing diffuse irradiance
        """
        if sh_degree > 5:
            raise NotImplementedError
        else:
            self.sh_degree = sh_degree
        self.sh_dim = (sh_degree +1)**2 
        self.base = base.squeeze()
        # Define constant attributes for diffuse irradiance computation
        self.NUM_CHANNELS = 3
        self.C1 = 0.429043
        self.C2 = 0.511664
        self.C3 = 0.743125
        self.C4 = 0.886227
        self.C5 = 0.247708
        self._FG_LUT = torch.as_tensor(np.fromfile('scene/NVDIFFREC/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')


    def clone(self):
        return EnvironmentLight(self.base.clone().detach())

    @property
    def get_shdim(self):
        return self.sh_dim

    @property
    def get_shdegree(self):
        return self.sh_degree

    @property
    def get_base(self):
        return self.base


    def set_base(self, base: torch.Tensor):
        assert base.squeeze().shape[0] == self.sh_dim, f"The number of SH coefficients must be {self.sh_dim}"
        self.base = base.squeeze()


    def get_diffuse_irradiance(self, normal: torch.tensor)-> torch.tensor:
        """
        The function computes the diffuse irradiance according to section 3.2 of "An efficient representaiton for Irradiance Environment Maps"
        by Ramamoorthi and Pat Hanrahan, https://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf. The implementation
        follows LumiGauss implementation https://arxiv.org/abs/2408.04474.
        The diffuse irradiance is computed by convolving environment light and cosine term in frequency domain. In the
        SH expansion of the environment light only terms up to degree 2 are considered.

        Args:
            normal: tensor of shape (N,3) containing normal vectors in R³.
        Returns:
            diffuse_irradiance: tensor of shape (N,1) containing the diffuse irradiance for each normal vector.
        """

        x, y, z = normal[..., 0, None], normal[..., 1, None], normal[..., 2, None]

        diffuse_irradiance = (
            self.C1 * self.base[8,:] * (x ** 2 - y ** 2) +
            self.C3 * self.base[6,:] * (z ** 2) +
            self.C4 * self.base[0,:] -
            self.C5 * self.base[6,:] +
            2 * self.C1 * self.base[4,:]* x * y +
            2 * self.C1 * self.base[7,:] * x * z +
            2 * self.C1 * self.base[5,:] * y * z +
            2 * self.C2 * self.base[3,:]* x +
            2 * self.C2 * self.base[1,:] * y +
            2 * self.C2 * self.base[2,:] * z
        )

        return diffuse_irradiance


    def get_specular_light_sh(self, kr: torch.Tensor)->torch.tensor:
        """
        The function computes specular lighting SH coefficients by convolving
        envionment light and a Gaussian blur kernel of std = kr in frequency domain.
        For what concerns the Gaussian blur filter its representation in frequency domain,
        the Gauss-Weierstrass kernel, is used to derive the corresponding SH coefficients.

        Args: 
            kr: roughness tensor of shape N x 1 containing.
        Returns:
            spec_light: tensor of shape N x self.sh_dim x 3 storing the SH coefficients
                                       of specular light for each roughness value and channel.
        """
        # Build coefficients of blur kernel in frequency (SH) domain
        gwk_sh = gauss_kernel(kr, self.sh_degree) # N x 25
        gwk_sh = gwk_sh.unsqueeze(-1) # N x 25 x 1
        # Adjust dimensions
        envlight_sh = self.base.unsqueeze(0)   # 1 x 25 x 3
        envlight_sh = envlight_sh.repeat(gwk_sh.shape[0], 1, 1) # N x 25 x 3
        # Perform convolution
        spec_light = gwk_sh * envlight_sh # N x 25 x 3

        return spec_light


    def sample_illumination(self, gb_pos:torch.tensor, view_pos: torch.tensor):
        dir = util.safe_normalize(gb_pos - view_pos).squeeze()
        illu_hdr = torch.nn.functional.relu(eval_sh(self.sh_degree,
                                                       self.base.unsqueeze(0).expand(dir.shape[0], -1, -1).transpose(1,2),
                                                       dir))

        return util.gamma_correction(illu_hdr) # linear --> sRGB


    def shade(self, gb_pos:torch.tensor, gb_normal:torch.tensor, albedo:torch.tensor, view_pos:torch.tensor,
              kr:torch.tensor=None, km:torch.tensor=None, specular:bool=True)->Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
       The function, based on NVDIFFREC implementation https://github.com/NVlabs/nvdiffrecmc,
        returns the emitted radiance in the input outgoing direction. 
        If specular is True a Microfacets Cook-Torrane reflectivity model is assumed, otherwise the model is assumed to be Lambertian. 
        In the specular case the final radiance is the sum of the diffuse and specular radiances.
        Args:
            gb_pos: world positions HxWxNx3
            gb_normal: normal vectors HxWxNx3
            albedo : albedo of the surface, base color HxWxNx3
            kr:roguhness of points HxWxNx1
            km: metalness of points HxWxNx1
            view_pos: camera position HxWxNx3
            envlight: SH coefficients of environment light 1xself.sh_dimx3
        Returns:
            rgb: shaded rgb color of shape HxWxNx3.
            extras: dictionary storing diffuse and specular radiance.
        """        

        nrmvec = gb_normal

        diffuse_irradiance_hdr = torch.clamp_min(self.get_diffuse_irradiance(nrmvec.squeeze()), 1e-4)
        # Compute diffuse color
        diffuse_rgb_hdr = albedo * diffuse_irradiance_hdr
        # Gamma correction: linear --> sRGB
        diffuse_rgb_ldr = util.gamma_correction(diffuse_rgb_hdr)
        extras = {"diffuse": diffuse_rgb_ldr}

        if not specular:
            extras.update({"specular": torch.zeros_like(extras["diffuse"])})

            return diffuse_rgb_ldr, extras
        else:
            wo = util.safe_normalize(view_pos - gb_pos) # (H, W, N, 3)
            reflvec = util.safe_normalize(util.reflect(wo, gb_normal))
            # Lookup FG term from lookup texture
            NdotV = torch.clamp(util.dot(wo, nrmvec), min=1e-4)
            fg_uv = torch.cat((NdotV, kr), dim=-1)
            fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')
            # Convovlve base SH coeffs with SH coeffs of Gaussian kernel
            spec_light = self.get_specular_light_sh(kr.squeeze([0,1])) # (N, 25, 3)
            # Compute specular irradiance in reflection direction
            spec_irradiance_hdr = eval_sh(self.sh_degree, spec_light.transpose(1,2), reflvec.squeeze())
            # Adjust dimensions
            spec_irradiance_hdr = torch.clamp_min(spec_irradiance_hdr[None, None, ...], 1e-4) # (H, W, N, 3)
            # Compute Fresnel-Schlick reflectivity
            if km is None:
                F0 = torch.ones_like(albedo) * 0.04  # [1, H, W, 3]
            else:
                F0 = (1.0 - km) * 0.04 + albedo * km
            reflectivity = F0 * fg_lookup[...,0:1] + fg_lookup[...,1:2]
            # Compute specular color
            specular_rgb_hdr = spec_irradiance_hdr * reflectivity
            if km is None:
                shaded_rgb = diffuse_rgb_hdr + specular_rgb_hdr
            else:
                shaded_rgb = (1 - km) * diffuse_rgb_hdr + specular_rgb_hdr
            # Gamma correction: linear --> sRGB
            shaded_rgb = util.gamma_correction(shaded_rgb)
            extras.update({'specular': util.gamma_correction(specular_rgb_hdr)})
            
            return shaded_rgb, extras


    def render_sh(self, width: int = 600)->torch.tensor:
        """Render environment light SH coefficients in equirectangular format"""
        self.base = self.base.squeeze()
        if isinstance(self.base, torch.Tensor):     
            rendered_sh = torch.tensor(sh_render(self.base.cpu().numpy(), width = width))
        else:
            rendered_sh = torch.tensor(sh_render(self.base, width = width))
        rendered_sh = util.gamma_correction(rendered_sh)
        return rendered_sh



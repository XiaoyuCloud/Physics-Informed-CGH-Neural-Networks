from . import colour_functions as cf
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from pathlib import Path
from PIL import Image

import numpy as np
from .backend_functions import backend as bd

import tensorflow as tf
import torch
from keras import backend as K
import cupy

m = 1.
cm = 1e-2
mm = 1e-3
um = 1e-6
nm = 1e-9
W = 10



class MonochromaticField:
    def __init__(self,  wavelength, extent_x, extent_y, Nx, Ny, intensity = 0.1 * W / (m**2)):
        """
        Initializes the field, representing the cross-section profile of a plane wave

        Parameters
        ----------
        wavelength: wavelength of the plane wave
        extent_x: length of the rectangular grid 
        extent_y: height of the rectangular grid 
        Nx: horizontal dimension of the grid 
        Ny: vertical dimension of the grid 
        intensity: intensity of the field
        """
        global bd
        from .backend_functions import backend as bd

        self.extent_x = extent_x
        self.extent_y = extent_y

        self.x = self.extent_x*(bd.arange(Nx)-Nx//2)/Nx
        self.y = self.extent_y*(bd.arange(Ny)-Ny//2)/Ny
        self.xx, self.yy = bd.meshgrid(self.x, self.y)

        self.Nx = bd.int(Nx)
        self.Ny = bd.int(Ny)
        self.E = bd.ones((int(self.Ny), int(self.Nx))) * bd.sqrt(intensity)
        self.λ = wavelength
        self.z = 0
        self.cs = cf.ColourSystem(clip_method = 0)
        
    def add_rectangular_slit(self, x0, y0, width, height):
        """
        Creates a slit centered at the point (x0, y0) with width width and height height
        """
        t = bd.select(
            [
                ((self.xx > (x0 - width / 2)) & (self.xx < (x0 + width / 2)))
                & ((self.yy > (y0 - height / 2)) & (self.yy < (y0 + height / 2))),
                bd.ones_like(self.E, dtype=bool),
            ],
            [bd.ones_like(self.E), bd.zeros_like(self.E)],
        )
        self.E = self.E*t

        self.I = bd.real(self.E * bd.conjugate(self.E))  

    def add_circular_slit(self, x0, y0, R):
        """
        Creates a circular slit centered at the point (x0,y0) with radius R
        """

        t = bd.select(
            [(self.xx - x0) ** 2 + (self.yy - y0) ** 2 < R ** 2, bd.ones_like(self.E, dtype=bool)], [bd.ones_like(self.E), bd.zeros_like(self.E)]
        )

        self.E = self.E*t
        self.I = bd.real(self.E * bd.conjugate(self.E))  



    def add_gaussian_beam(self, w0):
        """
        Creates a Gaussian beam with radius equal to w0
        """

        r2 = self.xx**2 + self.yy**2 
        self.E = self.E*bd.exp(-r2/(w0**2))
        self.I = bd.real(self.E * bd.conjugate(self.E))  



    def add_diffraction_grid(self, D, a, Nx, Ny):
        """
        Creates a diffraction_grid with Nx *  Ny slits with separation distance D and width a
        """

        E0 = bd.copy(self.E)
        t = 0

        b = D - a
        width, height = Nx * a + (Nx - 1) * b, Ny * a + (Ny - 1) * b
        x0, y0 = -width / 2, height / 2

        x0 = -width / 2 + a / 2
        for _ in range(Nx):
            y0 = height / 2 - a / 2
            for _ in range(Ny):

                t += bd.select(
                    [
                        ((self.xx > (x0 - a / 2)) & (self.xx < (x0 + a / 2)))
                        & ((self.yy > (y0 - a / 2)) & (self.yy < (y0 + a / 2))),
                        bd.ones_like(self.E, dtype=bool),
                    ],
                    [bd.ones_like(self.E), bd.zeros_like(self.E)],)
                y0 -= D
            x0 += D
        self.E = self.E*t
        self.I = bd.real(self.E * bd.conjugate(self.E))  



    def add_aperture_from_image(self, path, image_size = None):
        """
        Load the image specified at "path" as a numpy graymap array. The imagen is centered on the plane and
        its physical size is specified in image_size parameter as image_size = (float, float)

        - If image_size isn't specified, the image fills the entire aperture plane
        """


        img = Image.open(Path(path))
        #imgGray=np.asarray(img) / 255.0
        img = img.convert("RGB")

        img_pixels_width, img_pixels_height = img.size

        if image_size != None:
            new_img_pixels_width, new_img_pixels_height = int(np.round(image_size[0] / self.extent_x  * self.Nx)),  int(np.round(image_size[1] / self.extent_y  * self.Ny))
        else:
            #by default, the image fills the entire aperture plane
            new_img_pixels_width, new_img_pixels_height = self.Nx, self.Ny

        img = img.resize((new_img_pixels_width, new_img_pixels_height))

        dst_img = Image.new("RGB", (self.Nx, self.Ny), "black" )
        dst_img_pixels_width, dst_img_pixels_height = dst_img.size

        Ox, Oy = (dst_img_pixels_width-new_img_pixels_width)//2, (dst_img_pixels_height-new_img_pixels_height)//2
        
        dst_img.paste( img , box = (Ox, Oy ))

        imgRGB = np.asarray(dst_img) / 255.0
        imgR = imgRGB[:, :, 0]
        imgG = imgRGB[:, :, 1]
        imgB = imgRGB[:, :, 2]
        t = 0.2990 * imgR + 0.5870 * imgG + 0.1140 * imgB
        #t = np.flip(t, axis = 0)

        self.E = self.E*bd.array(t)


        # compute Field Intensity
        self.I = bd.real(self.E * bd.conjugate(self.E))

    def add_aperture_from_array(self, array, image_size=None):
        """
        Load the image specified at "path" as a numpy graymap array. The imagen is centered on the plane and
        its physical size is specified in image_size parameter as image_size = (float, float)

        - If image_size isn't specified, the image fills the entire aperture plane
        """

        t = array
        #t = np.flip(t, axis=0)
        self.E = self.E * bd.array(t)

        # compute Field Intensity
        self.I = bd.real(self.E * bd.conjugate(self.E))

    def add_aperture_from_arrayTF(self, array, image_size=None):
        """
        Load the image specified at "path" as a numpy graymap array. The imagen is centered on the plane and
        its physical size is specified in image_size parameter as image_size = (float, float)

        - If image_size isn't specified, the image fills the entire aperture plane
        """

        t = array
        #t = np.flip(t, axis=0)
        self.E=t
        #self.E = self.E * bd.array(t)

        # compute Field Intensity
        self.I = tf.real(self.E * tf.conj(self.E))

    def add_lens(self, f, radius = None, aberration = None):
        """add a thin lens with a focal length equal to f """

        self.E = self.E * bd.exp(-1j*bd.pi/(self.λ*f) * (self.xx**2 + self.yy**2))

        if aberration != None:
            self.E = self.E*bd.exp(2*bd.pi * 1j *aberration(self.xx, self.yy))

        if radius != None:
            self.E = bd.where((self.xx**2 + self.yy**2) < radius**2, self.E, 0)



    def propagate(self, z):
        """compute the field in distance equal to z with the angular spectrum method"""

        self.z += z

        # compute angular spectrum
        fft_c = bd.fft.fft2(self.E)
        c = bd.fft.fftshift(fft_c)
        #c=fft_c
        c[0:round(self.Ny/2+1)]=0
        kx = 2*bd.pi*bd.fft.fftshift(bd.fft.fftfreq(self.Nx, d = self.x[1]-self.x[0]))
        ky = 2*bd.pi*bd.fft.fftshift(bd.fft.fftfreq(self.Ny, d = self.y[1]-self.y[0]))
        kx, ky = bd.meshgrid(kx, ky)

        argument = (2 * bd.pi / self.λ) ** 2 - kx ** 2 - ky ** 2

        #Calculate the propagating and the evanescent (complex) modes
        tmp = bd.sqrt(bd.abs(argument))
        kz = bd.where(argument >= 0, tmp, 1j*tmp)

        # propagate the angular spectrum a distance z
        E = bd.fft.ifft2(bd.fft.ifftshift(c * bd.exp(1j * kz * z)))
        #E = bd.fft.ifft2(c * bd.exp(1j * kz * z))
        self.E = E

        # compute Field Intensity
        self.I = bd.real(E * bd.conjugate(E))

    def propagateTF(self, z):
        """compute the field in distance equal to z with the angular spectrum method"""

        self.z += z

        # compute angular spectrum
        E_complex = tf.cast(self.E, tf.complex128)
        # fft_c = tf.signal.fft2d(E_complex)
        # c = tf.signal.fftshift(fft_c)
        c=tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(E_complex)))
        #c=fft_c
        filter0=np.zeros((round(self.Ny / 2 + 1),round(self.Nx)))
        filter1 = np.ones((round(self.Ny / 2 - 1), round(self.Nx)))
        filter = torch.from_numpy(np.vstack((filter0, filter1)))
        c= tf.multiply(c,filter)
        kx = 2 * bd.pi * bd.fft.fftshift(bd.fft.fftfreq(self.Nx, d=self.x[1] - self.x[0]))
        ky = 2 * bd.pi * bd.fft.fftshift(bd.fft.fftfreq(self.Ny, d=self.y[1] - self.y[0]))
        kx, ky = bd.meshgrid(kx, ky)

        argument = (2 * bd.pi / self.λ) ** 2 - kx ** 2 - ky ** 2

        # Calculate the propagating and the evanescent (complex) modes
        # tmp = bd.sqrt(bd.abs(argument))
        # kz = bd.where(argument >= 0, tmp, 1j * tmp)
        kz=bd.sqrt(argument)
        # propagate the angular spectrum a distance z
        g = cupy.asnumpy(bd.exp(1j * kz * z))
        g=torch.from_numpy(g)
        E=tf.signal.fftshift(tf.signal.ifft2d(tf.signal.fftshift(tf.multiply(c,g))))
        self.E = E

        # compute Field Intensity
        self.I = tf.math.real(E * tf.math.conj(E))

    def get_colors(self):
        """ compute RGB colors"""

        rgb = self.cs.wavelength_to_sRGB(self.λ / nm, 10 * self.I.flatten()).T.reshape(
            (self.Ny, self.Nx, 3)
        )
        return rgb


    def compute_colors_at(self, z):
        """propagate the field to a distance equal to z and compute the RGB colors of the beam profile profile"""

        self.propagate(z)
        rgb = self.get_colors()
        return rgb

    def compute_colors_atTF(self, z):
        """propagate the field to a distance equal to z and compute the RGB colors of the beam profile profile"""

        self.propagateTF(z)
        return self.I

    def plot_intensity(self, square_root = False, figsize=(7, 6), xlim=None, ylim=None):
        """visualize the diffraction pattern with matplotlib"""

        plt.style.use("dark_background")


        if square_root == False:
            if bd != np:
                I = self.I.get()
            else:
                I = self.I

        else:
            if bd != np:
                I = np.sqrt(self.I.get())
            else:
                I = np.sqrt(self.I)


        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        if xlim != None:
            ax.set_xlim(xlim)

        if ylim != None:
            ax.set_ylim(ylim)

        # we use mm by default
        ax.set_xlabel("[mm]")
        ax.set_ylabel("[mm]")

        ax.set_title("Screen distance = " + str(self.z * 100) + " cm")

        im = ax.imshow(
            I, cmap= 'inferno',
            extent=[
                -self.extent_x / 2 / mm,
                self.extent_x / 2 / mm,
                -self.extent_y / 2 / mm,
                self.extent_y / 2 / mm,
            ],
            interpolation="spline36", origin = "lower"
        )

        cb = fig.colorbar(im, orientation = 'vertical')

        if square_root == False:
            cb.set_label(r'Intensity $\left[W / m^2 \right]$', fontsize=13, labelpad =  14 )
        else:
            cb.set_label(r'Square Root Intensity $\left[ \sqrt{W / m^2 } \right]$', fontsize=13, labelpad =  14 )


        plt.show()



    def plot(self, rgb, figsize=(9, 16), xlim=None, ylim=None):
        """visualize the diffraction pattern with matplotlib"""

        plt.style.use("dark_background")
        if bd != np:
            rgb = rgb.get()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        if xlim != None:
            ax.set_xlim(xlim)

        if ylim != None:
            ax.set_ylim(ylim)

        # we use mm by default
        ax.set_xlabel("[mm]")
        ax.set_ylabel("[mm]")

        ax.set_title("Screen distance = " + str(self.z * 100) + " cm")

        im = ax.imshow(
            (rgb),
            extent=[
                -self.extent_x / 2 / mm,
                self.extent_x / 2 / mm,
                -self.extent_y / 2 / mm,
                self.extent_y / 2 / mm,
            ],
            interpolation="spline36", origin = "upper"
        , vmin=0, vmax=1)
        plt.show()


    def add_spatial_noise(self, noise_radius, f_mean, f_size, N = 30, A = 1):
        """
        add spatial noise following a radial normal distribution

        Parameters
        ----------
        noise_radius: maximum radius affected by the spatial noise
        f_mean: mean spatial frequency of the spatial noise 
        f_size: spread spatial frequency of the noise 
        N: number of samples
        A: amplitude of the noise
        """

        def random_noise(xx,yy, f_mean,A):
            A = bd.random.rand(1)*A
            phase = bd.random.rand(1)*2*bd.pi
            fangle = bd.random.rand(1)*2*bd.pi
            f = bd.random.normal(f_mean, f_size/2)

            fx = f*bd.cos(fangle) 
            fy = f*bd.sin(fangle) 
            return A*bd.exp((xx**2 + yy**2)/ (noise_radius*2)**2)*bd.sin(2*bd.pi*fx*xx + 2*bd.pi*fy*yy + phase)

        E_noise = 0
        for i in range(0,N):
            E_noise += random_noise(self.xx,self.yy,f_mean,A)/bd.sqrt(N)

        self.E += E_noise *bd.exp(-(self.xx**2 + self.yy**2)/ (noise_radius)**2)
        self.I = bd.real(self.E * bd.conjugate(self.E)) 

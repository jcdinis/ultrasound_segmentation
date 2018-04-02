# -*- coding: utf-8*-
import math
import multiprocessing
import plistlib
import sys
import tempfile

import constants as const
import wx
import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import vtk
from vtk.wx.wxVTKRenderWindowInteractor import wxVTKRenderWindowInteractor
from vtk.util import numpy_support
from dcm2mmap import dcm2mmap, make_mask, dcmmf2mmap, dcmmf2mmapVTI
from skimage.filters import threshold_otsu

from wx.lib.pubsub import setuparg1

import wx.lib.pubsub.pub as Publisher
from wx.lib.pubsub import pub

from PIL import Image, ImageDraw
from scipy import stats

from skimage.draw import (line, polygon, circle,
                          circle_perimeter,
                          ellipse, ellipse_perimeter,
                          bezier_curve)
import Filtros
import morphsnakes

from skimage.feature import greycomatrix, greycoprops

#import smooth_c
from smooth_c import smooth
import floodfill
from FloatSliderText import FloatSliderText

from scipy import stats

from skimage.morphology import watershed
from skimage.feature import peak_local_max



NUMPY_TO_VTK_TYPE = {
                     'int16': 'SetScalarTypeToShort',
                     'uint16': 'SetScalarTypeToUnsignedShort',
                     'uint8': 'SetScalarTypeToUnsignedChar',
                     'float32': 'SetScalarTypeToFloat'
                    }

SIZE_KERNEL = 20

wildcard = "VTK image file (*.vti)|*.vti|" \
            "All files (*.*)|*.*"






def get_LUT_value(data, window, level):
    """Apply the RGB Look-Up Table for the given data and window/level value."""
    return np.piecewise(data,
                        [data <= (level - 0.5 - (window - 1) / 2),
                         data > (level - 0.5 + (window - 1) / 2)],
                        [0, 255, lambda data: ((data - (level - 0.5)) / (window - 1) + 0.5) * (255 - 0)])


def traca_poligono(pontos, shape):
    print ">>>>" , pontos
    print shape
    img = numpy.zeros(shape)
    poly=numpy.array(pontos)
    rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)
    img[rr, cc] = 1

    return img


def faz_level_set(img, poligono, it):
    gmin, gmax = 0.0, 1.0
    fmin, fmax = img.min(), img.max()

    nimg = ((gmax - gmin)/(fmax - fmin))*(img - fmin) + gmin

    gI = morphsnakes.gborders(nimg, alpha=1, sigma=2)

    # Morphological GAC. Initialization of the level-set.
    mgac = morphsnakes.MorphGAC(gI, smoothing=2, threshold=0.3, balloon=-1)
    mgac.levelset = poligono

    for i in xrange(it):
        mgac.step()

        yield mgac.levelset
        # manda exibir a mascara
        # mgac.levelset


def faz_level_set_3d(img, poligono, it):
    gmin, gmax = 0.0, 1.0
    fmin, fmax = img.min(), img.max()

    nimg = ((gmax - gmin)/(fmax - fmin))*(img - fmin) + gmin

    macwe = morphsnakes.MorphACWE(nimg, smoothing=3, lambda1=1, lambda2=1)
    #macwe= morphsnakes.MorphGAC(nimg,smoothing=1,threshold=0.6, balloon=-2)
    macwe.levelset = poligono

    for i in xrange(it):
        macwe.step()

        yield macwe.levelset
        # manda exibir a mascara
        # mgac.levelset


def to_vtk(n_array, spacing):
    dz, dy, dx = n_array.shape
    n_array.shape = dx * dy * dz

    print n_array.shape

    v_image = numpy_support.numpy_to_vtk(n_array)

    # Generating the vtkImageData
    image = vtk.vtkImageData()
    image.SetDimensions(dx, dy, dz)
    image.SetOrigin(0, 0, 0)
    image.SetSpacing(spacing)
    #image.SetNumberOfScalarComponents(1)
    image.SetExtent(0, dx -1, 0, dy -1, 0, dz - 1)
    #image.SetScalarType(numpy_support.get_vtk_array_type(n_array.dtype))
    image.AllocateScalars(numpy_support.get_vtk_array_type(n_array.dtype),1)
    image.GetPointData().SetScalars(v_image)
    #image.Update()

    image_copy = vtk.vtkImageData()
    image_copy.DeepCopy(image)
    #image_copy.Update()

    n_array.shape = dz, dy, dx
    return image_copy

# widget que com um slider e um campo de texto
class SliderText(wx.Panel):
    def __init__(self, parent, id, value, Min, Max):
        wx.Panel.__init__(self, parent, id)
        self.min = Min
        self.max = Max
        self.value = value
        self.build_gui()
        self.__bind_events_wx()
        self.Show()

    def build_gui(self):
        self.sliderctrl = wx.Slider(self, -1, self.value, self.min, self.max)
        self.textbox = wx.TextCtrl(self, -1, "%d" % self.value)


        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.sliderctrl, 1, wx.EXPAND)
        sizer.Add(self.textbox, 0, wx.EXPAND)
        self.SetSizer(sizer)

        self.Layout()
        self.Update()
        self.SetAutoLayout(1)

    def __bind_events_wx(self):
        self.sliderctrl.Bind(wx.EVT_SCROLL, self.do_slider)
        self.Bind(wx.EVT_SIZE, self.onsize)

    def onsize(self, evt):
        print "OnSize"
        evt.Skip()

    def do_slider(self, evt):
        self.value =  self.sliderctrl.GetValue()
        self.textbox.SetValue("%d" % self.value)
        evt.Skip()

    def GetValue(self):
        return self.value


class Window(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, size=(500, 500))
        self.build_gui()
        #self.run_image()
        self.__bind_events()
        self.__bind_events_wx()
        self.Show()

        self._texture_img = None

    def build_gui(self):
        self.axial_viewer = Viewer(self, 'AXIAL')
        self.coronal_viewer = Viewer(self, 'CORONAL')
        self.sagital_viewer = Viewer(self, 'SAGITAL')
        self.cortical_viewer = Viewer3D(self, '3D Model')
        
        self.calculo_density1=wx.Button(self,-1,"Apply Filter")
        self.reset_filter=wx.Button(self,-1,"Reset Filter")
        self.apaga_mask=wx.Button(self,-1,"Mask Delete")
        self.reset_mask=wx.Button(self,-1,"Reset mask")
        self.Plot_roi=wx.Button(self,-1,"3D Plot")
        #escolher a funcao para calculo da densidade
        self.autores_densidade=wx.ComboBox(self, -1, "Gaussian Gradiente de Magnitude", choices=("Gaussian Gradiente de Magnitude", "Sobel", "Correlate","Prewitt","Bilateral","Lapaciono","Fourie_Gaussian","Morphological_gradient","Wiener","Canny","ansisss","mediana","treshold_adptive","Lee filter","slic","Opening","sharpening"),
                                    style=wx.CB_READONLY)
        self.sampleList = ['Axial Plan', 'Coronal Plan','Sagital Plan']
        self.rb = wx.RadioBox(
                self, -1, "Choose the draw polygon plan", wx.DefaultPosition, wx.DefaultSize,
                self.sampleList, 2, wx.RA_SPECIFY_COLS
                )
        self.select_region=wx.CheckBox(self,-1, label='Select')

        self.interation_number= wx.SpinCtrl(self, -1,value='0',size=(5,-1))
        self.interation_number.SetRange(1, 600)
        self.cumprimento_tr= wx.SpinCtrl(self, -1,value='0',size=(5,-1))
        self.cumprimento_tr.SetRange(1,100)


        self.processing_snake_2d=wx.Button(self,-1,"Process Level Let 2D")
        self.processing_snake_3d=wx.Button(self,-1,"Process Level Set 3D")
        self.percentage_between_pixel=FloatSliderText(self, -1, '%', 0.01, 0.01, 1, 0.01)

        self.percentage_accept=FloatSliderText(self, -1, '%', 0.01, 0.01, 1, 0.01)

        self.processing_region_growing=wx.Button(self,-1,"Region Growing Auto_threshold")
        self.threshold_alto=FloatSliderText(self, -1, 'T1', 0.0, -1000.0, 2550.0, 1.0)
        self.threshold_baixo=FloatSliderText(self, -1, 'T0', 0.0, -1000.0, 2550.0, 1.0)
        self.processing_region_growing_controle_thresholding=wx.Button(self,-1,"Region Growing controle_threshold")
        self.processing_Smoothing=wx.Button(self,-1,"Process Smoothing surface")

        #escolher a funcao para calculo do modulo de elasticidade

        # regiao de corte

        viewer_sizer = wx.BoxSizer(wx.HORIZONTAL)
        viewer_sizer.Add(self.axial_viewer, 1, wx.EXPAND|wx.GROW)
        viewer_sizer.Add(self.coronal_viewer, 1, wx.EXPAND|wx.GROW)
        viewer_sizer.Add(self.sagital_viewer, 1, wx.EXPAND|wx.GROW)
        viewer_densidade_sizer = wx.BoxSizer(wx.HORIZONTAL)
        viewer_densidade_sizer.Add(self.cortical_viewer, 1, wx.EXPAND|wx.GROW)

        b_sizer = wx.BoxSizer(wx.VERTICAL)

        viewers_sizer = wx.BoxSizer(wx.VERTICAL)
        viewers_sizer.Add(viewer_sizer, 10, wx.EXPAND)
        viewers_sizer.Add(viewer_densidade_sizer, 10, wx.EXPAND)


        b_sizer.Add(wx.StaticText(self, -1, "Type of Filters") , 0, wx.EXPAND)
        b_sizer.Add(self.autores_densidade)

        b_sizer.Add(wx.StaticText(self, -1, "Filter Processing") , 0, wx.EXPAND)
        b_sizer.Add(self.calculo_density1)

        b_sizer.Add(wx.StaticText(self, -1, "Reset Filter") , 0, wx.EXPAND)
        b_sizer.Add(self.reset_filter)

        b_sizer.Add(wx.StaticText(self, -1, "Mask Delete") , 0, wx.EXPAND)
        b_sizer.Add(self.apaga_mask)
        b_sizer.Add(self.reset_mask)

        b_sizer.Add(wx.StaticText(self, -1) , 0, wx.EXPAND)
        b_sizer.Add(self.rb,0)

        b_sizer.Add(wx.StaticText(self, -1, u"Selected Region of Interests") , 0, wx.EXPAND)
        b_sizer.Add(self.select_region)

        b_sizer.Add(wx.StaticText(self, -1, u"Number of Interation") , 0, wx.EXPAND)
        b_sizer.Add(self.interation_number, 0, wx.EXPAND)

        b_sizer.Add(wx.StaticText(self, -1, u"Processing Level Let") , 0, wx.EXPAND)
        b_sizer.Add(self.processing_snake_2d)

        b_sizer.Add(wx.StaticText(self, -1, u'Length depth mask') , 0, wx.EXPAND)
        b_sizer.Add(self.cumprimento_tr, 0, wx.EXPAND)
        b_sizer.Add(self.processing_snake_3d)

        b_sizer.Add(wx.StaticText(self, -1,u"Percentage of acceptance among the pixel neighborhood"),0,wx.EXPAND)
        b_sizer.Add(self.percentage_between_pixel,0, wx.EXPAND)

        b_sizer.Add(wx.StaticText(self, -1, u"Processing with Region Growing") , 0, wx.EXPAND)
        b_sizer.Add(self.processing_region_growing)


        b_sizer.Add(wx.StaticText(self, -1,u"Low and upper threshold"),0,wx.EXPAND)
        b_sizer.Add(self.threshold_baixo,0, wx.EXPAND)
        b_sizer.Add(self.threshold_alto,0, wx.EXPAND)

        b_sizer.Add(wx.StaticText(self, -1, u"Processing with Region Growing_controle") , 0, wx.EXPAND)
        b_sizer.Add(self.processing_region_growing_controle_thresholding)

        b_sizer.Add(wx.StaticText(self, -1,u"Percentage of acceptance"),0,wx.EXPAND)
        b_sizer.Add(self.percentage_accept,0, wx.EXPAND)


        


        b_sizer.Add(wx.StaticText(self, -1, u"3D Plot The Region Of Interese") , 0, wx.EXPAND)
        b_sizer.Add(self.Plot_roi)

        b_sizer.Add(wx.StaticText(self, -1, u"Smoothing a 3D surface and replot") , 0, wx.EXPAND)
        b_sizer.Add(self.processing_Smoothing)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(viewers_sizer, 10, wx.EXPAND)
        sizer.Add(b_sizer, 3, wx.EXPAND)

        #button cut matriz

        self.SetSizer(sizer)


        ############################# MENU ##################
        menubar = wx.MenuBar()
        fileMenu = wx.Menu()
        fitem = fileMenu.Append(wx.ID_EXIT, 'Quit', 'Quit application')
        mOpen=fileMenu.Append(-1, 'Abrir Dicom','Abrir Dicom' )
        mOpen2=fileMenu.Append(-1, 'Abrir Dicom Mult-frame','Abrir Dicom Mult-frame' )
        mOpen3=fileMenu.Append(-1, 'Abrir VTI','Abrir VTI' )

        msave = fileMenu.Append(-1, 'Save Mask', 'Save Mask')
        isave = fileMenu.Append(-1, 'Save Image', 'Save Image')
        stlsave = fileMenu.Append(-1, 'Save Stl', 'Save Stl')

        menubar.Append(fileMenu, '&File')
        self.SetMenuBar(menubar)


        self.Bind(wx.EVT_MENU, self.OnQuit, fitem)
        self.Bind(wx.EVT_MENU, self.mOpenImagem, mOpen)
        self.Bind(wx.EVT_MENU, self.mOpenImagem2, mOpen2)
        self.Bind(wx.EVT_MENU, self.mOpenImagem3, mOpen3)

        self.Bind(wx.EVT_MENU, self.OnSaveMask, msave)
        self.Bind(wx.EVT_MENU, self.OnSaveImage, isave)
        self.Bind(wx.EVT_MENU, self.OnSaveSTL, stlsave)

        ##########################################################

        self.Layout()
        self.Update()
        self.SetAutoLayout(1)

    def mOpenImagem(self, evt):
        dialog = wx.DirDialog(None, "Choose a directory:",style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON)
        if dialog.ShowModal() == wx.ID_OK:
            receberImagems=dialog.GetPath().encode("latin1")
            print receberImagems
            self.run_image(receberImagems)
        dialog.Destroy()

    def run_image(self, input_dir):
        self.image_file = tempfile.mktemp()
        m_input, self.spacing = dcm2mmap(input_dir, self.image_file)
        self.mask_file = tempfile.mktemp()
        mask = make_mask(m_input, self.mask_file)
        mask[:] = 0
        mask[0, :, :] = 1
        mask[:, 0, :] = 1
        mask[:, :, 0] = 1
        self.axial_viewer.SetInput(m_input, mask, self.spacing)
        self.sagital_viewer.SetInput(m_input, mask, self.spacing)
        self.coronal_viewer.SetInput(m_input, mask, self.spacing)

        self.markers = numpy.zeros(m_input.shape, 'int8')

        self.m_input = m_input

        self.m_mask = mask

        self.original_image = numpy.memmap(tempfile.mktemp(),
            shape=self.m_input.shape,
            dtype=self.m_input.dtype,
            mode='w+')
        self.original_image[:] = self.m_input


    def mOpenImagem2(self, evt):
        dialog = wx.FileDialog(None, "Choose a directory:",style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON)
        if dialog.ShowModal() == wx.ID_OK:
            receberImagems=dialog.GetPath().encode("latin1")
            print receberImagems
            self.run_image2(receberImagems)
        dialog.Destroy()

    def run_image2(self, input_dir):
        self.image_file = tempfile.mktemp()
        m_input, self.spacing = dcmmf2mmap(input_dir, self.image_file)
        self.mask_file = tempfile.mktemp()
        mask = make_mask(m_input, self.mask_file)
        mask[:] = 0
        mask[0, :, :] = 1
        mask[:, 0, :] = 1
        mask[:, :, 0] = 1
        self.axial_viewer.SetInput(m_input, mask, self.spacing)
        self.sagital_viewer.SetInput(m_input, mask, self.spacing)
        self.coronal_viewer.SetInput(m_input, mask, self.spacing)

        self.markers = numpy.zeros(m_input.shape, 'int8')

        self.m_input = m_input

        self.m_mask = mask
        print m_input.shape

        self.original_image = numpy.memmap(tempfile.mktemp(),
            shape=self.m_input.shape,
            dtype=self.m_input.dtype,
            mode='w+')
        self.original_image[:] = self.m_input




    def mOpenImagem3(self, evt):
        dialog = wx.FileDialog(None, "Choose a directory:",style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON)
        if dialog.ShowModal() == wx.ID_OK:
            receberImagems=dialog.GetPath().encode("latin1")
            print receberImagems
            self.run_image3(receberImagems)
        dialog.Destroy()

    def run_image3(self, input_dir):
        self.image_file = tempfile.mktemp()
        m_input, self.spacing = dcmmf2mmapVTI(input_dir, self.image_file)
        self.mask_file = tempfile.mktemp()
        mask = make_mask(m_input, self.mask_file)
        mask[:] = 0
        mask[0, :, :] = 1
        mask[:, 0, :] = 1
        mask[:, :, 0] = 1
        self.axial_viewer.SetInput(m_input, mask, self.spacing)
        self.sagital_viewer.SetInput(m_input, mask, self.spacing)
        self.coronal_viewer.SetInput(m_input, mask, self.spacing)

        self.markers = numpy.zeros(m_input.shape, 'int8')

        self.m_input = m_input

        self.m_mask = mask
        print m_input.shape

        self.original_image = numpy.memmap(tempfile.mktemp(),
            shape=self.m_input.shape,
            dtype=self.m_input.dtype,
            mode='w+')
        self.original_image[:] = self.m_input

    def __bind_events(self):
        Publisher.subscribe(self.add_marker,
                                 'Add marker')


    def __bind_events_wx(self):

        self.calculo_density1.Bind(wx.EVT_BUTTON,self.calculo_density)
        self.reset_filter.Bind(wx.EVT_BUTTON, self.reset_image)
        self.select_region.Bind(wx.EVT_CHECKBOX, self.estado_seleciona)
        self.processing_snake_2d.Bind(wx.EVT_BUTTON, self.processa_tudo_2d)
        self.processing_snake_3d.Bind(wx.EVT_BUTTON, self.processa_tudo_3d)
        self.processing_region_growing.Bind(wx.EVT_BUTTON, self.processa_region_growing)

        self.apaga_mask.Bind(wx.EVT_BUTTON, self.del_marked_mask)
        self.reset_mask.Bind(wx.EVT_BUTTON, self.resetar_mask)

        self.processing_region_growing_controle_thresholding.Bind(wx.EVT_BUTTON, self.processa_region_growing_controle_thresolding)

        self.Plot_roi.Bind(wx.EVT_BUTTON, self.chamar3dplot)
        self.processing_Smoothing.Bind(wx.EVT_BUTTON, self.chamarSmoothing)

    def _dialog_save(self):
        """
        Create and show the Save FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Save file as ...",
            defaultDir='',
            defaultFile="", wildcard=wildcard, style=wx.SAVE
            )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
        else:
            path = ''
        dlg.Destroy()

        return path

    def OnSaveMask(self, e):
        path = self._dialog_save()
        if path:
            mask = numpy.array(self.m_mask[1:, 1:, 1:])
            vtkimg = to_vtk(mask, self.spacing)

            w = vtk.vtkXMLImageDataWriter()
            w.SetFileName(path)
            w.SetInputData(vtkimg)
            w.Write()

    def OnSaveImage(self, e):
        path = self._dialog_save()
        if path:
            vtkimg = to_vtk(self.m_input, self.spacing)

            w = vtk.vtkXMLImageDataWriter()
            w.SetFileName(path)
            w.SetInputData(vtkimg)
            w.Write()


    def OnSaveSTL(self, e):
        path = self._dialog_save()
        if path:
            w = vtk.vtkSTLWriter()
            w.SetFileTypeToBinary()
            w.SetFileName(path)
            w.SetInputData(self.cortical_viewer.actor.GetMapper().GetInput())
            w.Write()

    def estado_seleciona(self, evt):
        pub.sendMessage('Altera estado seleciona vertices', self.select_region.GetValue())



    def reset_image(self, evt):
        self.m_input[:] = self.original_image

        na = self.axial_viewer.scroll.GetThumbPosition()
        nc = self.coronal_viewer.scroll.GetThumbPosition()
        ns = self.sagital_viewer.scroll.GetThumbPosition()

        self.axial_viewer.SetInput(self.m_input, self.m_mask, self.spacing)
        self.coronal_viewer.SetInput(self.m_input, self.m_mask, self.spacing)
        self.sagital_viewer.SetInput(self.m_input, self.m_mask, self.spacing)

        self.axial_viewer.scroll.SetThumbPosition(na)
        self.coronal_viewer.scroll.SetThumbPosition(nc)
        self.sagital_viewer.scroll.SetThumbPosition(ns)

        self.axial_viewer.SetSliceNumber(na)
        self.coronal_viewer.SetSliceNumber(nc)
        self.sagital_viewer.SetSliceNumber(ns)



    def del_marked_mask(self, evt):
        image = self.m_input
        mask = self.coronal_viewer.mask[1:, 1:, 1:]
        
        image[mask > 127] = image.min()
        
        self.coronal_viewer.mask[:] = 1
        mask[:] = 0

        self.m_mask[:] = self.coronal_viewer.mask

        na = self.axial_viewer.scroll.GetThumbPosition()
        nc = self.coronal_viewer.scroll.GetThumbPosition()
        ns = self.sagital_viewer.scroll.GetThumbPosition()

        self.axial_viewer.SetInput(self.m_input, self.m_mask, self.spacing)
        self.coronal_viewer.SetInput(self.m_input, self.m_mask, self.spacing)
        self.sagital_viewer.SetInput(self.m_input, self.m_mask, self.spacing)

        self.axial_viewer.scroll.SetThumbPosition(na)
        self.coronal_viewer.scroll.SetThumbPosition(nc)
        self.sagital_viewer.scroll.SetThumbPosition(ns)

        self.axial_viewer.SetSliceNumber(na)
        self.coronal_viewer.SetSliceNumber(nc)
        self.sagital_viewer.SetSliceNumber(ns)

        self.axial_viewer.list_points_ff = []
        self.coronal_viewer.list_points_ff = []
        self.sagital_viewer.list_points_ff = []

    def resetar_mask(self, evt):
        self.coronal_viewer.mask[:] = 1
        mask = self.coronal_viewer.mask[1:, 1:, 1:]
        mask[:] = 0

        self.m_mask[:] = self.coronal_viewer.mask

        self.axial_viewer.mask[:] = self.m_mask
        self.coronal_viewer.mask[:] = self.m_mask
        self.sagital_viewer.mask[:] = self.m_mask

        na = self.axial_viewer.scroll.GetThumbPosition()
        nc = self.coronal_viewer.scroll.GetThumbPosition()
        ns = self.sagital_viewer.scroll.GetThumbPosition()

        self.axial_viewer.scroll.SetThumbPosition(na)
        self.coronal_viewer.scroll.SetThumbPosition(nc)
        self.sagital_viewer.scroll.SetThumbPosition(ns)

        self.axial_viewer.SetSliceNumber(na)
        self.coronal_viewer.SetSliceNumber(nc)
        self.sagital_viewer.SetSliceNumber(ns)


    def add_marker(self, pubsub_evt):
        position, value = pubsub_evt.data
        self.markers[position] = value
        print "marker added", position, value


    def processa_tudo_2d(self, evt):
        if self.rb.GetSelection() == 1:
            points = self.coronal_viewer.list_points
            pol = traca_poligono(points, self.m_input[:, 0, :].shape)

            its = self.interation_number.GetValue()
            n = self.coronal_viewer.scroll.GetThumbPosition()

            mask = self.coronal_viewer.mask[1:, n+1,1:]
            mask[:] = pol.astype('uint8') * 255

            img = self.coronal_viewer.image_input[:, n, :]

            for m in faz_level_set(img, pol, its):
                mask[:] = m.astype('uint8') * 255
                self.coronal_viewer.SetSliceNumber(n)


        elif self.rb.GetSelection() == 2:
            print 'hhhhhhhhhhsed'
            points = self.sagital_viewer.list_points
            pol = traca_poligono(points, self.m_input[:, :, 0].shape)

            #plt.imshow(pol); plt.show()

            its = self.interation_number.GetValue()
            n = self.sagital_viewer.scroll.GetThumbPosition()

            mask = self.sagital_viewer.mask[1:,1:,n+1]
            mask[:] = pol.astype('uint8') * 255

            img = self.sagital_viewer.image_input[:, :, n]

            for m in faz_level_set(img, pol, its):
                mask[:] = m.astype('uint8') * 255
                self.sagital_viewer.SetSliceNumber(n)


        elif self.rb.GetSelection() == 0:
            points = self.axial_viewer.list_points
            pol = traca_poligono(points, self.m_input[0, :, :].shape)

            its = self.interation_number.GetValue()
            n = self.axial_viewer.scroll.GetThumbPosition()

            mask = self.axial_viewer.mask[n+1,1:,1:]
            mask[:] = pol.astype('uint8') * 255

            img = self.axial_viewer.image_input[n, :, :]

            for m in faz_level_set(img, pol, its):
                mask[:] = m.astype('uint8') * 255
                self.axial_viewer.SetSliceNumber(n)


    def processa_tudo_3d(self, evt):
        if self.rb.GetSelection() == 1:
            points = self.coronal_viewer.list_points
            pol = traca_poligono(points, self.m_input[:, 0, :].shape)
            its = self.interation_number.GetValue()
            n = self.coronal_viewer.scroll.GetThumbPosition()
            mask = self.coronal_viewer.mask[1:, 1:, 1:]
            mask[:] = 0

            ct=self.cumprimento_tr.GetValue()

            for i in xrange(n-ct,n+ct+1):
                mask[:, i, :] = pol.astype('uint8') * 255
            img = self.coronal_viewer.image_input
            #mask_cp = (mask > 150) * 1
            mask_cp = (mask > 200) * 1

            #print pol.shape, img.shape, mask.shape
            for m in faz_level_set_3d(img, mask_cp, its):
                mask[:] = m.astype('uint8') * 255
                self.coronal_viewer.SetSliceNumber(n)

        if self.rb.GetSelection() == 2:
            points = self.sagital_viewer.list_points
            pol = traca_poligono(points, self.m_input[:, :, 0].shape)
            its = self.interation_number.GetValue()
            n = self.sagital_viewer.scroll.GetThumbPosition()
            mask = self.sagital_viewer.mask[1:, 1:, 1:]
            mask[:] = 0
            ct=self.cumprimento_tr.GetValue()

            for i in xrange(n-ct,n+ct+1):
                mask[:, :, i] = pol.astype('uint8') * 255
            img = self.sagital_viewer.image_input
            #mask_cp = (mask > 150) * 1
            mask_cp = (mask > 200) * 1

            #print pol.shape, img.shape, mask.shape
            for m in faz_level_set_3d(img, mask_cp, its):
                mask[:] = m.astype('uint8') * 255
                self.sagital_viewer.SetSliceNumber(n)

        if self.rb.GetSelection() == 0:
            points = self.axial_viewer.list_points
            pol = traca_poligono(points, self.m_input[0, :, :].shape)
            its = self.interation_number.GetValue()
            n = self.axial_viewer.scroll.GetThumbPosition()
            mask = self.axial_viewer.mask[1:, 1:, 1:]
            mask[:] = 0
            ct=self.cumprimento_tr.GetValue()

            for i in xrange(n-ct,n+ct+1):
                mask[i, :, :] = pol.astype('uint8') * 255
            img = self.axial_viewer.image_input
            mask_cp = (mask > 200) * 1
            #mask_cp = (mask > 150) * 1

            #print pol.shape, img.shape, mask.shape
            for m in faz_level_set_3d(img, mask_cp, its):
                mask[:] = m.astype('uint8') * 255
                self.axial_viewer.SetSliceNumber(n)



    def processa_region_growing(self, evt):
        if self.rb.GetSelection() == 1:
            points = self.coronal_viewer.list_points_ff

        if self.rb.GetSelection() == 2:
            points = self.sagital_viewer.list_points_ff

        elif self.rb.GetSelection() == 0:
            points = self.axial_viewer.list_points_ff


        e1 = self.percentage_between_pixel.GetValue()
        mask = self.sagital_viewer.mask
        mask[:] = 0

        floodfill.floodfill_auto_threshold(self.m_input, points, e1, 255, mask[1:, 1:, 1:])

        mask[0, :, :] = 1
        mask[:, 0, :] = 1
        mask[:, :, 0] = 1

        self.axial_viewer.mask[:] = mask
        self.coronal_viewer.mask[:] = mask
        self.sagital_viewer.mask[:] = mask

        self.axial_viewer.SetSliceNumber(self.axial_viewer.scroll.GetThumbPosition())
        self.coronal_viewer.SetSliceNumber(self.coronal_viewer.scroll.GetThumbPosition())
        self.sagital_viewer.SetSliceNumber(self.sagital_viewer.scroll.GetThumbPosition())


    def processa_region_growing_controle_thresolding(self, evt):
        if self.rb.GetSelection() == 1:
            points = self.coronal_viewer.list_points_ff

        if self.rb.GetSelection() == 2:
            points = self.sagital_viewer.list_points_ff

        elif self.rb.GetSelection() == 0:
            points = self.axial_viewer.list_points_ff


        t0=self.threshold_baixo.GetValue()
        t1=self.threshold_alto.GetValue()


        mask = self.sagital_viewer.mask
        mask[:] = 0

        floodfill.floodfill_threshold(self.m_input, points, t0,t1, 255, mask[1:, 1:, 1:])

        mask[0, :, :] = 1
        mask[:, 0, :] = 1
        mask[:, :, 0] = 1

        self.axial_viewer.mask[:] = mask
        self.coronal_viewer.mask[:] = mask
        self.sagital_viewer.mask[:] = mask

        self.axial_viewer.SetSliceNumber(self.axial_viewer.scroll.GetThumbPosition())
        self.coronal_viewer.SetSliceNumber(self.coronal_viewer.scroll.GetThumbPosition())
        self.sagital_viewer.SetSliceNumber(self.sagital_viewer.scroll.GetThumbPosition())



    def do_watershed(self,evt):
    	if self.rb.GetSelection() == 1:
            points = self.coronal_viewer.list_points_ff

        if self.rb.GetSelection() == 2:
            points = self.sagital_viewer.list_points_ff

        elif self.rb.GetSelection() == 0:
            points = self.axial_viewer.list_points_ff


        markers = np.zers_like(self.m_input, dtype='uint8')








    


    def chamar3dplot(self,evt):
        mask = numpy.array(self.m_mask[1:, 1:, 1:])

        mask_in = numpy.zeros([i + 2 for i in mask.shape], dtype='uint8')
        mask_in[1:-1, 1:-1, 1:-1] = mask
        vtkimg = to_vtk(mask_in, self.spacing)
        self.cortical_viewer.SetInput3D(vtkimg)

    def chamarSmoothing(self,evt):
        self.smoothing()


    def OnQuit(self, e):
        self.Destroy()



    def calculo_density(self,evt):
        print self.m_input.shape
        if self.autores_densidade.GetValue()=="Gaussian Gradiente de Magnitude":
            Matriz_filtrado = Filtros.Filtro1(self.m_input)

        elif self.autores_densidade.GetValue()=="Sobel":
            Matriz_filtrado = Filtros.Filtro2(self.m_input)

        elif self.autores_densidade.GetValue()=="Correlate":
            Matriz_filtrado = Filtros.Filtro3(self.m_input)

        elif self.autores_densidade.GetValue()=="Prewitt":
            Matriz_filtrado = Filtros.Filtro4(self.m_input)

        elif self.autores_densidade.GetValue()=="Bilateral":
            Matriz_filtrado = Filtros.Filtro5(self.m_input)

        elif self.autores_densidade.GetValue()=="Lapaciono":
            Matriz_filtrado = Filtros.Filtro6(self.m_input)


        elif self.autores_densidade.GetValue()=="Fourie_Gaussian":
            Matriz_filtrado = Filtros.Filtro7(self.m_input)


        elif self.autores_densidade.GetValue()=="Morphological_gradient":
            Matriz_filtrado = Filtros.Filtro8(self.m_input)

        elif self.autores_densidade.GetValue()=="Wiener":
            Matriz_filtrado = Filtros.Filtro9(self.m_input)

        elif self.autores_densidade.GetValue()=="Canny":
            Matriz_filtrado = Filtros.Filtro10(self.m_input)

        elif self.autores_densidade.GetValue()=="ansisss":
            Matriz_filtrado = Filtros.anisodiff3(self.m_input)

        elif self.autores_densidade.GetValue()=="mediana":
            Matriz_filtrado = Filtros.Filtro11(self.m_input)


        elif self.autores_densidade.GetValue()=="treshold_adptive":
            Matriz_filtrado = Filtros.Filtro12(self.m_input)


        elif self.autores_densidade.GetValue()=="Lee filter":
            Matriz_filtrado = Filtros.Filtro13(self.m_input)

        elif self.autores_densidade.GetValue()=="slic":
            Matriz_filtrado = Filtros.Filtro14(self.m_input)

        elif self.autores_densidade.GetValue()=="Opening":
            Matriz_filtrado = Filtros.Filtro_opening(self.m_input)

        elif self.autores_densidade.GetValue()=="sharpening":
            Matriz_filtrado =Filtros.sharpening(self.m_input)



        mask = self.m_mask
        gmin, gmax = 0, 255 #self.m_input.min(), self.m_input.max()
        fmin, fmax = Matriz_filtrado.min(), Matriz_filtrado.max()

        matrizf = (float(gmax - gmin)/float(fmax - fmin)) * (Matriz_filtrado - fmin) + gmin

        print matrizf.min(), matrizf.max()

        na = self.axial_viewer.scroll.GetThumbPosition()
        nc = self.coronal_viewer.scroll.GetThumbPosition()
        ns = self.sagital_viewer.scroll.GetThumbPosition()

        self.axial_viewer.SetInput(matrizf, mask, self.spacing)
        self.axial_viewer.scroll.SetThumbPosition(na)
        self.axial_viewer.SetSliceNumber(na)

        self.coronal_viewer.SetInput(matrizf, mask, self.spacing)
        self.coronal_viewer.scroll.SetThumbPosition(nc)
        self.coronal_viewer.SetSliceNumber(nc)

        self.sagital_viewer.SetInput(matrizf, mask, self.spacing)
        self.sagital_viewer.scroll.SetThumbPosition(ns)
        self.sagital_viewer.SetSliceNumber(ns)
        print matrizf.shape, Matriz_filtrado.shape
        self.m_input[:] = matrizf

    def smoothing(self):
        its = self.interation_number.GetValue()
        print self.m_mask.max()
        mask = numpy.array((self.m_mask[1:, 1:, 1:] > 127)*1, dtype='uint8')
        print mask.max(), its
        out_img=smooth.smoothpy(mask, its)

        mask_in = numpy.zeros([i + 2 for i in out_img.shape], dtype='float64')
        mask_in[:] = -1
        mask_in[1:-1, 1:-1, 1:-1] = out_img

        vtkimg = to_vtk(mask_in, self.spacing)
        self.cortical_viewer.SetInput3D(vtkimg, 0.0)



class Viewer(wx.Panel):

    def __init__(self, prnt, orientation='AXIAL', tipo='NORMAL', titulo=None):
        wx.Panel.__init__(self, prnt)

        self.orientation = orientation
        self.slice_number = 0
        self.tipo = tipo

        self.actor = None

        self.image_input = None

        if titulo is None:
            self.titulo =  self.orientation
        else:
            self.titulo = titulo

        self.__init_gui()
        self.config_interactor()
        self.__bind_events_wx()
 
        self.msize = 1
        self.mtype = "Max"

        self.ww = 255
        self.wl = 127

    def __init_gui(self):
        interactor = wxVTKRenderWindowInteractor(self, -1, size=self.GetSize())

        scroll = wx.ScrollBar(self, -1, style=wx.SB_VERTICAL)
        self.scroll = scroll

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.StaticText(self, -1, self.titulo))
        sizer.Add(interactor, 1, wx.EXPAND|wx.GROW)

        background_sizer = wx.BoxSizer(wx.HORIZONTAL)
        background_sizer.AddSizer(sizer, 1, wx.EXPAND|wx.GROW|wx.ALL, 2)
        background_sizer.Add(scroll, 0, wx.EXPAND|wx.GROW)
        self.SetSizer(background_sizer)
        background_sizer.Fit(self)

        self.Layout()
        self.Update()
        self.SetAutoLayout(1)

        self.interactor = interactor

        self.pega_pontos = False
        self.continua_pegar = True

        self.list_points=[]
        self.list_points_ff = []
        self.list_points_ff_other = []

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.picker=vtk.vtkPointPicker()

        pub.subscribe(self.altera_estado, 'Altera estado seleciona vertices')
        


    def __update_camera(self):
        orientation = self.orientation

        self.cam.SetFocalPoint(0, 0, 0)
        self.cam.SetViewUp(const.SLICE_POSITION[1][0][self.orientation])
        self.cam.SetPosition(const.SLICE_POSITION[1][1][self.orientation])
        self.cam.ParallelProjectionOn()


    def config_interactor(self):
        style = vtk.vtkInteractorStyleImage()
        self.style = style

        self.picker = vtk.vtkCellPicker()
        self.actor = None

        ren = vtk.vtkRenderer()
        ren.SetBackground((0, 0, 0))
        #ren.AddActor(self.actor)
        ren.SetLayer(0)

        self.ren = ren

        cam = ren.GetActiveCamera()
        cam.ParallelProjectionOn()

        ren2 = vtk.vtkRenderer()
        ren2.SetInteractive(0)

        ren2.SetActiveCamera(cam)
        ren2.SetLayer(1)

        interactor = self.interactor
        interactor.SetInteractorStyle(style)
        interactor.GetRenderWindow().SetNumberOfLayers(2)
        interactor.GetRenderWindow().AddRenderer(ren)
        interactor.GetRenderWindow().AddRenderer(ren2)
        interactor.SetPicker(self.picker)


        style.AddObserver("LeftButtonPressEvent",self.Clik)
        style.AddObserver("LeftButtonReleaseEvent", self.Release)


        self.cam = ren.GetActiveCamera()
        self.ren = ren

        self.linhaActor1=vtk.vtkActor()
        self.linhaActor1.GetProperty().SetRepresentationToWireframe()
        self.ren.AddActor(self.linhaActor1)



    def AddMarker(self, x, y, z):
        s = vtk.vtkSphereSource()
        s.SetCenter(x, y, z)
        s.SetRadius(self.spacing[0])

        m = vtk.vtkGlyph3DMapper()
        m.SetInputConnection(s.GetOutputPort())
        m.OrientOn()

        a = vtk.vtkActor()
        a.SetMapper(m)

        self.ren.AddActor(a)

    def altera_estado(self, pubsub_evt):
        if self.continua_pegar:
            print 'ggggggg', pubsub_evt.data
            self.pega_pontos = pubsub_evt.data
            if self.list_points:
                self.list_points.append(self.list_points[0])
                self.continua_pegar = False

                print self.list_points

    def Clik(self,obj, evt):
        print 'derttttt'
        if self.pega_pontos:
            iren =self.interactor
            ren = self.ren
            x, y = iren.GetEventPosition()
            self.x0 = x
            self.y0 = y
            print x


    def Release(self, obj, event):
        if self.pega_pontos:
            iren =self.interactor
            ren = self.ren
            x, y = iren.GetEventPosition()
            self.x1 = x
            self.y1 = y


            self.picker.Pick(x, y, 0, ren)
            p_position = self.picker.GetPickPosition()
            position = self.actor.GetInput().FindPoint(p_position)

            if self.orientation == 'SAGITAL':
                print 'SAGTIAL', self.image.GetDimensions()
                py = position/self.image.GetDimensions()[1]
                px = position%self.image.GetDimensions()[1]
            else:
                py = position/self.image.GetDimensions()[0]
                px = position%self.image.GetDimensions()[0]

            pz = self.scroll.GetThumbPosition()

            self.list_points.append((py,px))

            if iren.GetControlKey():
                if self.orientation == 'AXIAL':
                    self.list_points_ff_other.append((px, py, pz))
                elif self.orientation == 'CORONAL':
                    self.list_points_ff_other.append((px, pz, py))
                elif self.orientation == 'SAGITAL':
                    self.list_points_ff_other.append((pz, px, py))
            else:
                if self.orientation == 'AXIAL': 
                    self.list_points_ff.append((px, py, pz))
                elif self.orientation == 'CORONAL':
                    self.list_points_ff.append((px, pz, py))
                elif self.orientation == 'SAGITAL':
                    self.list_points_ff.append((pz, px, py))

            print self.list_points


    def SetOrientation(self, orientation):
        self.orientation = orientation
        if self.orientation == 'AXIAL':
            max_slice_number = len(self.image_input)
        elif self.orientation == 'CORONAL':
            max_slice_number = self.image_input.shape[1]
        elif self.orientation == 'SAGITAL':
            max_slice_number = self.image_input.shape[2]

        self.scroll.SetScrollbar(wx.SB_VERTICAL, 1, max_slice_number,
                                 max_slice_number)
        self.__update_camera()
        self.ren.ResetCamera()



    def SetInput(self, m_input, mask, spacing, tipo='NORMAL'):
        self.image_input = m_input
        self.mask = mask
        self.spacing = spacing
        self.tipo=tipo
        if self.orientation == 'AXIAL':
            max_slice_number = len(m_input)
        elif self.orientation == 'CORONAL':
            max_slice_number = m_input.shape[1]
        elif self.orientation == 'SAGITAL':
            max_slice_number = m_input.shape[2]


        self.scroll.SetScrollbar(wx.SB_VERTICAL, 1, max_slice_number,
                                 max_slice_number)
        cam = self.ren.GetActiveCamera()
        actor = vtk.vtkActor()
        self.ren.AddActor(actor)
        print "Added CUBE"

        self.SetSliceNumber(0)
        self.__update_camera()
        self.ren.ResetCamera()



    def __bind_events_wx(self):
        self.scroll.Bind(wx.EVT_SCROLL, self.OnScrollBar)





    def OnScrollBar(self, evt):
        n = self.scroll.GetThumbPosition()
        print "Slice ->", n
        self.SetSliceNumber(n)

    def add_marker(self, obj, evt):
        mouse_x, mouse_y = self.interactor.GetEventPosition()
        self.picker.Pick(mouse_x, mouse_y, 0, self.ren)
        p_position = self.picker.GetPickPosition()
        position = self.actor.GetInput().FindPoint(p_position)
        n = self.scroll.GetThumbPosition()
        value = 255 if evt == 'LeftButtonPressEvent' else 0
        Publisher.sendMessage('Add marker', ((n,
                                                   position/self.image.GetDimensions()[0],
                                                   position%self.image.GetDimensions()[1]),
                                                 value))

    def SetSliceNumber(self, n):
        if self.image_input is None:
            return
        print self.orientation, n, self.msize, self.mtype
        if self.orientation == 'AXIAL':
                n_array = numpy.array(self.image_input[n])

                mask = self.mask[n+1]

        elif self.orientation == 'CORONAL':
                n_array = numpy.array(self.image_input[:, n, :])

                mask = self.mask[1:, n+1, 1:]

        elif self.orientation == 'SAGITAL':
                n_array = numpy.array(self.image_input[:, : ,n])

                mask = numpy.array(self.mask[1:, 1:, n+1])

        n_shape = n_array.shape

        image = self.to_vtk(n_array, self.spacing, n, self.orientation)
        vmask = self.to_vtk(mask, self.spacing, n, self.orientation)

        print "O TIPO EH", self.tipo


        self.image = self.do_ww_wl(image)

        self.image = self.do_blend(self.image, self.do_colour_mask(vmask))

        if self.actor is None:
            self.actor = vtk.vtkImageActor()
            self.actor.PickableOn()
            self.actor.SetInputData(self.image)
            self.ren.AddActor(self.actor)
        else:
            self.actor.SetInputData(self.image)

        self.actor.SetDisplayExtent(self.image.GetExtent())

        self.__update_display_extent(self.image)
        self.interactor.Render()



    def __update_display_extent(self, image):
        self.actor.SetDisplayExtent(image.GetExtent())
        self.ren.ResetCameraClippingRange()

    def to_vtk(self, n_array, spacing, slice_number, orientation):
        try:
            dz, dy, dx = n_array.shape
        except ValueError:
            dy, dx = n_array.shape
            dz = 1

        v_image = numpy_support.numpy_to_vtk(n_array.flat)

        if orientation == 'AXIAL':
            extent = (0, dx -1, 0, dy -1, slice_number, slice_number + dz - 1)
        elif orientation == 'SAGITAL':
            dx, dy, dz = dz, dx, dy
            extent = (slice_number, slice_number + dx - 1, 0, dy - 1, 0, dz - 1)
        elif orientation == 'CORONAL':
            dx, dy, dz = dx, dz, dy
            extent = (0, dx - 1, slice_number, slice_number + dy - 1, 0, dz - 1)

        # Generating the vtkImageData
        image = vtk.vtkImageData()
        image.SetOrigin(0, 0, 0)
        image.SetSpacing(spacing)
        #image.SetNumberOfScalarComponents(1)
        image.SetDimensions(dx, dy, dz)
        image.SetExtent(extent)
        #image.SetScalarType(numpy_support.get_vtk_array_type(n_array.dtype))
        image.AllocateScalars(numpy_support.get_vtk_array_type(n_array.dtype), 1)
        image.GetPointData().SetScalars(v_image)
        #image.Update()

        image_copy = vtk.vtkImageData()
        image_copy.DeepCopy(image)
        #image_copy.Update()

        return image_copy


    def do_ww_wl(self, image):
        colorer = vtk.vtkImageMapToWindowLevelColors()
        colorer.SetInputData(image)
        colorer.SetWindow(self.ww)
        colorer.SetLevel(self.wl)
        colorer.SetOutputFormatToRGB()
        colorer.Update()

        return colorer.GetOutput()

    def do_blend(self, imagedata, mask):
        # blend both imagedatas, so it can be inserted into viewer
        print "Blending Spacing", imagedata.GetSpacing(), mask.GetSpacing()

        blend_imagedata = vtk.vtkImageBlend()
        blend_imagedata.SetBlendModeToNormal()
        blend_imagedata.SetOpacity(1, 0.8)
        blend_imagedata.SetInputData(imagedata)
        blend_imagedata.AddInputData(mask)
        blend_imagedata.Update()


        # return colorer.GetOutput()

        return blend_imagedata.GetOutput()

    def __create_background(self, imagedata):

        thresh_min, thresh_max = imagedata.GetScalarRange()

        # map scalar values into colors
        lut_bg = vtk.vtkLookupTable()
        lut_bg.SetTableRange(thresh_min, thresh_max)
        lut_bg.SetSaturationRange(0, 0)
        lut_bg.SetHueRange(0, 0)
        lut_bg.SetValueRange(0, 1)
        lut_bg.Build()

        # map the input image through a lookup table
        img_colours_bg = vtk.vtkImageMapToColors()
        img_colours_bg.SetOutputFormatToRGBA()
        img_colours_bg.SetLookupTable(lut_bg)
        img_colours_bg.SetInputData(imagedata)
        img_colours_bg.Update()

        return img_colours_bg.GetOutput()

    def do_blend(self, imagedata, mask):
        # blend both imagedatas, so it can be inserted into viewer
        print "Blending Spacing", imagedata.GetSpacing(), mask.GetSpacing()

        blend_imagedata = vtk.vtkImageBlend()
        blend_imagedata.SetBlendModeToNormal()
        # blend_imagedata.SetOpacity(0, 1.0)
        blend_imagedata.SetOpacity(1, 0.8)
        blend_imagedata.SetInputData(imagedata)
        blend_imagedata.AddInputData(mask)
        blend_imagedata.Update()


        # return colorer.GetOutput()

        return blend_imagedata.GetOutput()


    def do_colour_mask(self, imagedata):
        scalar_range = int(imagedata.GetScalarRange()[1])
        r,g,b = 0, 1, 0

        # map scalar values into colors
        lut_mask = vtk.vtkLookupTable()
        lut_mask.SetNumberOfColors(255)
        lut_mask.SetHueRange(const.THRESHOLD_HUE_RANGE)
        lut_mask.SetSaturationRange(1, 1)
        lut_mask.SetValueRange(0, 1)
        lut_mask.SetNumberOfTableValues(256)
        lut_mask.SetTableValue(0, 0, 0, 0, 0.0)
        lut_mask.SetTableValue(1, 0, 0, 0, 0.0)
        lut_mask.SetTableValue(2, 0, 0, 0, 0.0)
        lut_mask.SetTableValue(255, r, g, b, 1.0)
        lut_mask.SetRampToLinear()
        lut_mask.Build()
        # self.lut_mask = lut_mask

        # map the input image through a lookup table
        img_colours_mask = vtk.vtkImageMapToColors()
        img_colours_mask.SetLookupTable(lut_mask)
        img_colours_mask.SetOutputFormatToRGBA()
        img_colours_mask.SetInputData(imagedata)
        img_colours_mask.Update()
        # self.img_colours_mask = img_colours_mask

        return img_colours_mask.GetOutput()



class Viewer3D(wx.Panel):

    def __init__(self, prnt,titulo):
        wx.Panel.__init__(self, prnt)

        self.renderer = vtk.vtkRenderer()
        self.Interactor = wxVTKRenderWindowInteractor(self,-1, size = self.GetSize())
        self.Interactor.GetRenderWindow().AddRenderer(self.renderer)
        self.Interactor.Render()

        istyle = vtk.vtkInteractorStyleTrackballCamera()

        self.Interactor.SetInteractorStyle(istyle)

        self.actor = None

        hbox=wx.BoxSizer(wx.VERTICAL)
        hbox.Add(wx.StaticText(self,-1, u'Global'))


        hbox.Add(self.Interactor,1, wx.EXPAND)
        self.SetSizer(hbox)

        self.renderer.ResetCamera()

    def SetInput3D(self, image, value=127):
        if self.actor:
            self.renderer.RemoveActor(self.actor)

        mesh = vtk.vtkMarchingCubes()
        mesh.SetInputData(image)
        mesh.SetValue(0, value)
        mesh.Update()
        
        m = vtk.vtkPolyDataMapper()
        m.SetInputData(mesh.GetOutput())

        #cl=vtk.vtkFillHolesFilter()
        #cl.SetInput(m)

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(m)
        self.actor.GetProperty().SetColor(0.5,1,0.3)


        self.renderer.AddActor(self.actor)









class App(wx.App):
    def OnInit(self):
        self.frame = Window(None)
        self.frame.Center()
        self.SetTopWindow(self.frame)
        return True

if __name__ == '__main__':
    app = App(0)
    app.MainLoop()


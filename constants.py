#--------------------------------------------------------------------------
# Software:     InVesalius - Software de Reconstrucao 3D de Imagens Medicas
# Copyright:    (C) 2001  Centro de Pesquisas Renato Archer
# Homepage:     http://www.softwarepublico.gov.br
# Contact:      invesalius@cti.gov.br
# License:      GNU - GPL 2 (LICENSE.txt/LICENCA.txt)
#--------------------------------------------------------------------------
#    Este programa e software livre; voce pode redistribui-lo e/ou
#    modifica-lo sob os termos da Licenca Publica Geral GNU, conforme
#    publicada pela Free Software Foundation; de acordo com a versao 2
#    da Licenca.
#
#    Este programa eh distribuido na expectativa de ser util, mas SEM
#    QUALQUER GARANTIA; sem mesmo a garantia implicita de
#    COMERCIALIZACAO ou de ADEQUACAO A QUALQUER PROPOSITO EM
#    PARTICULAR. Consulte a Licenca Publica Geral GNU para obter mais
#    detalhes.
#--------------------------------------------------------------------------
import os

# Measurements

# VTK text
TEXT_SIZE_SMALL = 11
TEXT_SIZE = 12
TEXT_SIZE_LARGE = 16
TEXT_COLOUR = (1,1,1)

(X,Y) = (0.03, 0.97)
(XZ, YZ) = (0.05, 0.93)
TEXT_POS_LEFT_UP = (X, Y)
#------------------------------------------------------------------
TEXT_POS_LEFT_DOWN = (X, 1-Y) # SetVerticalJustificationToBottom

TEXT_POS_LEFT_DOWN_ZERO = (X, 1-YZ)
#------------------------------------------------------------------
TEXT_POS_RIGHT_UP = (1-X, Y) # SetJustificationToRight
#------------------------------------------------------------------
TEXT_POS_RIGHT_DOWN = (1-X, 1-Y) # SetVerticalJustificationToBottom &
                                 # SetJustificationToRight
#------------------------------------------------------------------
TEXT_POS_HCENTRE_DOWN = (0.5, 1-Y) # SetJustificationToCentered
                                   # ChildrticalJustificationToBottom

TEXT_POS_HCENTRE_DOWN_ZERO = (0.5, 1-YZ)
#------------------------------------------------------------------
TEXT_POS_HCENTRE_UP = (0.5, Y)  # SetJustificationToCentered
#------------------------------------------------------------------
TEXT_POS_VCENTRE_RIGHT = (1-X, 0.5) # SetVerticalJustificationToCentered
                                    # SetJustificationToRight
TEXT_POS_VCENTRE_RIGHT_ZERO = (1-XZ, 0.5)
#------------------------------------------------------------------
TEXT_POS_VCENTRE_LEFT = (X, 0.5) # SetVerticalJustificationToCentered
#------------------------------------------------------------------


# Slice orientation
AXIAL = 1
CORONAL = 2
SAGITAL = 3
VOLUME = 4
SURFACE = 5
DENSIDADE = 6

# Measure type
LINEAR = 6
ANGULAR = 7

# Colour representing each orientation
ORIENTATION_COLOUR = {'AXIAL': (1,0,0), # Red
                      'CORONAL': (0,1,0), # Green
                      'SAGITAL': (0,0,1)} # Blue


# Camera according to slice's orientation
#CAM_POSITION = {"AXIAL":(0, 0, 1), "CORONAL":(0, -1, 0), "SAGITAL":(1, 0, 0)}
#CAM_VIEW_UP =  {"AXIAL":(0, 1, 0), "CORONAL":(0, 0, 1), "SAGITAL":(0, 0, 1)}
AXIAL_SLICE_CAM_POSITION = {"AXIAL":(0, 0, 1), "CORONAL":(0, -1, 0), "SAGITAL":(1, 0, 0), "DENSIDADE":(0, 0, 1)}
AXIAL_SLICE_CAM_VIEW_UP =  {"AXIAL":(0, 1, 0), "CORONAL":(0, 0, 1), "SAGITAL":(0, 0, 1), "DENSIDADE":(0, 1, 0)}

SAGITAL_SLICE_CAM_POSITION = {"AXIAL":(0, 1, 0), "CORONAL":(1, 0, 0), "SAGITAL":(0, 0, -1), "DENSIDADE":(0, 1, 0)}
SAGITAL_SLICE_CAM_VIEW_UP =  {"AXIAL":(-1, 0, 0), "CORONAL":(0, 1, 0), "SAGITAL":(0, 1, 0), "DENSIDADE":(-1, 0, 0)}

CORONAL_SLICE_CAM_POSITION = {"AXIAL":(0, 1, 0), "CORONAL":(0, 0, 1), "SAGITAL":(1, 0, 0),"DENSIDADE":(0, 1, 0)}
CORONAL_SLICE_CAM_VIEW_UP =  {"AXIAL":(0, 0, -1), "CORONAL":(0, 1, 0), "SAGITAL":(0, 1, 0), "DENSIDADE":(0, 0, -1)}

SLICE_POSITION = {AXIAL:[AXIAL_SLICE_CAM_VIEW_UP, AXIAL_SLICE_CAM_POSITION],
                  SAGITAL:[SAGITAL_SLICE_CAM_VIEW_UP, SAGITAL_SLICE_CAM_POSITION],
                  CORONAL:[CORONAL_SLICE_CAM_VIEW_UP, CORONAL_SLICE_CAM_POSITION],
                  DENSIDADE:[AXIAL_SLICE_CAM_VIEW_UP, AXIAL_SLICE_CAM_POSITION]}
#Project Status
#NEW_PROJECT = 0
#OPEN_PROJECT = 1
#CHANGE_PROJECT = 2
#SAVE_PROJECT = 3
PROJ_NEW = 0
PROJ_OPEN = 1
PROJ_CHANGE = 2
PROJ_CLOSE = 3

PROJ_MAX = 4


####
MODE_RP = 0
MODE_NAVIGATOR = 1
MODE_RADIOLOGY = 2
MODE_ODONTOLOGY = 3






# Mask threshold options
THRESHOLD_PRESETS_INDEX = 0 #Bone
THRESHOLD_HUE_RANGE = (0, 0.6667)
THRESHOLD_INVALUE = 5000
THRESHOLD_OUTVALUE = 0

# Mask properties
MASK_OPACITY = 0.40
#MASK_OPACITY = 0.35
MASK_COLOUR =  [[0.33, 1, 0.33],
                [1, 1, 0.33],
                [0.33, 0.91, 1],
                [1, 0.33, 1],
                [1, 0.68, 0.33],
                [1, 0.33, 0.33],
                [0.33333333333333331, 0.33333333333333331, 1.0],
                #(1.0, 0.33333333333333331, 0.66666666666666663),
                [0.74901960784313726, 1.0, 0.0],
                [0.83529411764705885, 0.33333333333333331, 1.0]]#,
                #(0.792156862745098, 0.66666666666666663, 1.0),
                #(1.0, 0.66666666666666663, 0.792156862745098), # too "light"
                #(0.33333333333333331, 1.0, 0.83529411764705885),#],
                #(1.0, 0.792156862745098, 0.66666666666666663),
                #(0.792156862745098, 1.0, 0.66666666666666663), # too "light"
                #(0.66666666666666663, 0.792156862745098, 1.0)]


MEASURE_COLOUR =  [[1, 0, 0],
                [1, 0.4, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 0.6, 0]]

SURFACE_COLOUR =  [(0.33, 1, 0.33),
                (1, 1, 0.33),
                (0.33, 0.91, 1),
                (1, 0.33, 1),
                (1, 0.68, 0.33),
                (1, 0.33, 0.33),
                (0.33333333333333331, 0.33333333333333331, 1.0),
                (1.0, 0.33333333333333331, 0.66666666666666663),
                (0.74901960784313726, 1.0, 0.0),
                (0.83529411764705885, 0.33333333333333331, 1.0),
                (0.792156862745098, 0.66666666666666663, 1.0),
                (1.0, 0.66666666666666663, 0.792156862745098),
                (0.33333333333333331, 1.0, 0.83529411764705885),
                (1.0, 0.792156862745098, 0.66666666666666663),
                (0.792156862745098, 1.0, 0.66666666666666663),
                (0.66666666666666663, 0.792156862745098, 1.0)]

# Related to slice editor brush
BRUSH_CIRCLE = 0 #
BRUSH_SQUARE = 1
DEFAULT_BRUSH_FORMAT = BRUSH_CIRCLE

BRUSH_DRAW = 0
BRUSH_ERASE = 1
BRUSH_THRESH = 2
DEFAULT_BRUSH_OP = BRUSH_THRESH

BRUSH_COLOUR = (0,0,1.0)
BRUSH_SIZE = 30

# Surface creation values. Each element's list contains:
# 0: imagedata reformat ratio
# 1: smooth_iterations
# 2: smooth_relaxation_factor
# 3: decimate_reduction


REDUCE_IMAGEDATA_QUALITY = 0

ICON_DIR = os.path.abspath(os.path.join('..', 'icons'))
SAMPLE_DIR = os.path.abspath(os.path.join('..', 'samples'))
DOC_DIR = os.path.abspath(os.path.join('..', 'docs'))



# if 1, use vtkVolumeRaycastMapper, if 0, use vtkFixedPointVolumeRayCastMapper
TYPE_RAYCASTING_MAPPER = 0

folder=RAYCASTING_PRESETS_DIRECTORY= os.path.abspath(os.path.join("..",
                                                                  "presets",
                                                                  "raycasting"))



LOG_FOLDER = os.path.join(os.path.expanduser('~'), '.invesalius', 'logs')
if not os.path.isdir(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

folder = os.path.join(os.path.expanduser('~'), '.invesalius', 'presets')
if not os.path.isdir(folder):
    os.makedirs(folder)


USER_RAYCASTING_PRESETS_DIRECTORY = folder

# If 0 dont't blur, 1 blur
RAYCASTING_WWWL_BLUR = 0

RAYCASTING_PRESETS_FOLDERS = (RAYCASTING_PRESETS_DIRECTORY,
                              USER_RAYCASTING_PRESETS_DIRECTORY)


####
#MODE_ZOOM = 0 #"Set Zoom Mode",
#MODE_ZOOM_SELECTION = 1 #:"Set Zoom Select Mode",
#MODE_ROTATE = 2#:"Set Spin Mode",
#MODE_MOVE = 3#:"Set Pan Mode",
#MODE_WW_WL = 4#:"Bright and contrast adjustment"}
#MODE_LINEAR_MEASURE = 5


#        self.states = {0:"Set Zoom Mode", 1:"Set Zoom Select Mode",
#                       2:"Set Spin Mode", 3:"Set Pan Mode",
#                       4:"Bright and contrast adjustment"}


#ps.Publisher().sendMessage('Set interaction mode %d'%
#                                        (MODE_BY_ID[id]))

#('Set Editor Mode')
#{0:"Set Change Slice Mode"}

####
MODE_SLICE_SCROLL = -1
MODE_SLICE_EDITOR = -2
MODE_SLICE_CROSS = -3

############



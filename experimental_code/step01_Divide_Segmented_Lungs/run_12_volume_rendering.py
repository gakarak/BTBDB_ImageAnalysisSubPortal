import nibabel as nib
import vtk
import os
import numpy as np

lesion_id2rgb = {
    1: [1, 0, 0],
    2: [0, 1, 0],
    3: [0, 0, 1],
    4: [1, 1, 0],
    5: [0, 1, 1],
    6: [1, 0, 1],
    7: [0.7, 0.7, 0.7],
}


def vtk_show(renderer, width=400, height=300):
    """
    Takes vtkRenderer instance and returns an IPython Image with the rendering.
    """
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(width, height)
    renderWindow.Render()

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetWriteToMemory(1)
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()
    # data = str((writer.GetResult()))

    return


def createDummyRenderer():
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.9, 0.9, 0.9)

    camera = renderer.MakeCamera()
    camera.SetViewAngle(45.0)

    # camera.SetPosition(256, -512, 512)
    # camera.SetFocalPoint(0.0, 0.0, 255.0)
    # camera.SetViewUp(0.46, 0.80, -0.38)

    camera.SetPosition(256, 1256, 256)
    camera.SetFocalPoint(256, 256, 128)
    camera.SetViewUp(0, 0, -1)

    renderer.SetActiveCamera(camera)

    return renderer

if __name__ == '__main__':

    vol_opacity = 0.5

    root_dirname = '/home/snezhko/workspace/CRDF_6/btb_data/case-1790a000-8325-40ab-812b-ac08a89b1912/study-03eafb5b-994c-4c5b-8476-980d92f05900/'

    lesion_filename_ = root_dirname + 'series-1.2.840.113704.9.1000.16.1.2016100718154717100020002-CT-lesions4.nii.gz'
    lesion_prob_filename_ = root_dirname + 'series-1.2.840.113704.9.1000.16.1.2016100718154717100020002-CT-lesions4.nii.gz-prob.nii.gz'
    ct_filename_ = root_dirname + 'series-1.2.840.113704.9.1000.16.1.2016100718154717100020002-CT.nii.gz'
    lungs_filename_ = root_dirname + 'series-1.2.840.113704.9.1000.16.1.2016100718154717100020002-CT-lungs.nii.gz'

    # ct_img = nib.load(ct_filename_)
    # les_img = nib.load(lesion_filename_)
    # les_prob_img = nib.load(lesion_prob_filename_)
    # lungs_img = nib.load(lungs_filename_)
    if not os.path.isfile(lungs_filename_):
        print('no file {}'.format(lungs_filename_))
        exit()

    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(lesion_prob_filename_)

    cast_filter = vtk.vtkImageCast()
    cast_filter.SetInputConnection(reader.GetOutputPort())
    cast_filter.SetOutputScalarTypeToShort()
    cast_filter.Update()

    imdata_seg = cast_filter.GetOutput()

    func_color = vtk.vtkColorTransferFunction()
    for kk, vv in lesion_id2rgb.items():
        vv = np.array(vv)
        func_color.AddRGBPoint(kk, vv[0], vv[1], vv[2])

    func_opacity_scalar = vtk.vtkPiecewiseFunction()
    for kk, vv in lesion_id2rgb.items():
        func_opacity_scalar.AddPoint(kk, vol_opacity if kk > 0 else 0.5)

    func_opacity_gradient = vtk.vtkPiecewiseFunction()
    func_opacity_gradient.AddPoint(1, 0.0)
    func_opacity_gradient.AddPoint(5, 0.1)
    func_opacity_gradient.AddPoint(100, 1.0)

    volume_prop = vtk.vtkVolumeProperty()
    volume_prop.ShadeOn()
    volume_prop.SetColor(func_color)
    volume_prop.SetScalarOpacity(func_opacity_scalar)
    volume_prop.SetGradientOpacity(func_opacity_gradient)
    volume_prop.SetInterpolationTypeToLinear()
    volume_prop.ShadeOn()
    volume_prop.SetAmbient(0.4)
    volume_prop.SetDiffuse(1.0)
    volume_prop.SetSpecular(0.2)

    # func_ray_cast = vtk.vtkVolumeRayCastCompositeFunction()
    # func_ray_cast.SetCompositeMethodToClassifyFirst()

    volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
    # volume_mapper.SetVolumeRayCastFunction(func_ray_cast)
    volume_mapper.SetInputData(imdata_seg)

    volume_actor = vtk.vtkVolume()
    volume_actor.SetMapper(volume_mapper)
    volume_actor.SetProperty(volume_prop)

    renderer = createDummyRenderer()
    renderer.AddActor(volume_actor)

    # vtk_show(renderer, 600, 600)
    # exit()

    # renderer = vtk.vtkRenderer()
    # renderer.AddActor(volume_actor)


    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(2048, 2048)
    renderWindow.Render()

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(root_dirname + 'image.png')
    # writer.SetWriteToMemory(1)
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()
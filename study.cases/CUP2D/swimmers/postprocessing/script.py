# trace generated using paraview version 5.9.0-RC2

#### import the simple module from the paraview
from paraview.simple import *
import glob
import argparse

def ParaviewRendering(chi_files,tmp_files,file_name,max_frame,case):
	#### disable automatic camera reset on 'Show'
	paraview.simple._DisableFirstRenderCameraReset()
	
	# create a new 'XDMF Reader'
	chi_restart_0000000 = XDMFReader(registrationName='chi_restart_0000000*', FileNames=chi_files)
	chi_restart_0000000.CellArrayStatus = ['data']
	
	# get animation scene
	animationScene1 = GetAnimationScene()
	
	# update animation scene based on data timesteps
	animationScene1.UpdateAnimationUsingDataTimeSteps()
	
	# create a new 'XDMF Reader'
	tmp_restart_0000000 = XDMFReader(registrationName='tmp_restart_0000000*', FileNames=tmp_files)
	tmp_restart_0000000.CellArrayStatus = ['data']
	
	# Properties modified on tmp_restart_0000000
	tmp_restart_0000000.GridStatus = ['OctTree']
	
	# get active view
	renderView1 = GetActiveViewOrCreate('RenderView')
	
	# show data in view
	tmp_restart_0000000Display = Show(tmp_restart_0000000, renderView1, 'UniformGridRepresentation')
	
	# trace defaults for the display properties.
	tmp_restart_0000000Display.Representation = 'Outline'
	tmp_restart_0000000Display.ColorArrayName = ['CELLS', '']
	tmp_restart_0000000Display.SelectTCoordArray = 'None'
	tmp_restart_0000000Display.SelectNormalArray = 'None'
	tmp_restart_0000000Display.SelectTangentArray = 'None'
	tmp_restart_0000000Display.OSPRayScaleFunction = 'PiecewiseFunction'
	tmp_restart_0000000Display.SelectOrientationVectors = 'None'
	tmp_restart_0000000Display.ScaleFactor = 0.8250000000000001
	tmp_restart_0000000Display.SelectScaleArray = 'data'
	tmp_restart_0000000Display.GlyphType = 'Arrow'
	tmp_restart_0000000Display.GlyphTableIndexArray = 'data'
	tmp_restart_0000000Display.GaussianRadius = 0.04125
	tmp_restart_0000000Display.SetScaleArray = [None, '']
	tmp_restart_0000000Display.ScaleTransferFunction = 'PiecewiseFunction'
	tmp_restart_0000000Display.OpacityArray = [None, '']
	tmp_restart_0000000Display.OpacityTransferFunction = 'PiecewiseFunction'
	tmp_restart_0000000Display.DataAxesGrid = 'GridAxesRepresentation'
	tmp_restart_0000000Display.PolarAxes = 'PolarAxesRepresentation'
	tmp_restart_0000000Display.ScalarOpacityUnitDistance = 0.03673940176856459
	tmp_restart_0000000Display.OpacityArrayName = ['CELLS', 'data']
	tmp_restart_0000000Display.IsosurfaceValues = [0.0]
	tmp_restart_0000000Display.SliceFunction = 'Plane'
	tmp_restart_0000000Display.Slice = 34
	
	# init the 'Plane' selected for 'SliceFunction'
	tmp_restart_0000000Display.SliceFunction.Origin = [4.0, 2.0, 2.0]
	
	# reset view to fit data
	renderView1.ResetCamera()
	
	# get the material library
	materialLibrary1 = GetMaterialLibrary()
	
	# Properties modified on chi_restart_0000000
	chi_restart_0000000.GridStatus = ['OctTree']
	
	# show data in view
	chi_restart_0000000Display = Show(chi_restart_0000000, renderView1, 'UniformGridRepresentation')
	
	# trace defaults for the display properties.
	chi_restart_0000000Display.Representation = 'Outline'
	chi_restart_0000000Display.ColorArrayName = ['CELLS', '']
	chi_restart_0000000Display.SelectTCoordArray = 'None'
	chi_restart_0000000Display.SelectNormalArray = 'None'
	chi_restart_0000000Display.SelectTangentArray = 'None'
	chi_restart_0000000Display.OSPRayScaleFunction = 'PiecewiseFunction'
	chi_restart_0000000Display.SelectOrientationVectors = 'None'
	chi_restart_0000000Display.ScaleFactor = 0.8250000000000001
	chi_restart_0000000Display.SelectScaleArray = 'data'
	chi_restart_0000000Display.GlyphType = 'Arrow'
	chi_restart_0000000Display.GlyphTableIndexArray = 'data'
	chi_restart_0000000Display.GaussianRadius = 0.04125
	chi_restart_0000000Display.SetScaleArray = [None, '']
	chi_restart_0000000Display.ScaleTransferFunction = 'PiecewiseFunction'
	chi_restart_0000000Display.OpacityArray = [None, '']
	chi_restart_0000000Display.OpacityTransferFunction = 'PiecewiseFunction'
	chi_restart_0000000Display.DataAxesGrid = 'GridAxesRepresentation'
	chi_restart_0000000Display.PolarAxes = 'PolarAxesRepresentation'
	chi_restart_0000000Display.ScalarOpacityUnitDistance = 0.03673940176856459
	chi_restart_0000000Display.OpacityArrayName = ['CELLS', 'data']
	chi_restart_0000000Display.IsosurfaceValues = [0.49999991059303284]
	chi_restart_0000000Display.SliceFunction = 'Plane'
	chi_restart_0000000Display.Slice = 34
	
	# init the 'Plane' selected for 'SliceFunction'
	chi_restart_0000000Display.SliceFunction.Origin = [4.0, 2.0, 2.0]
	
	# update the view to ensure updated data information
	renderView1.Update()
	
	# set scalar coloring
	ColorBy(tmp_restart_0000000Display, ('FIELD', 'vtkBlockColors'))
	
	# show color bar/color legend
	tmp_restart_0000000Display.SetScalarBarVisibility(renderView1, True)
	
	# set scalar coloring
	ColorBy(chi_restart_0000000Display, ('FIELD', 'vtkBlockColors'))
	
	# show color bar/color legend
	chi_restart_0000000Display.SetScalarBarVisibility(renderView1, True)
	
	# get color transfer function/color map for 'vtkBlockColors'
	vtkBlockColorsLUT = GetColorTransferFunction('vtkBlockColors')
	
	# get opacity transfer function/opacity map for 'vtkBlockColors'
	vtkBlockColorsPWF = GetOpacityTransferFunction('vtkBlockColors')
	
	# create a new 'Merge Blocks'
	mergeBlocks1 = MergeBlocks(registrationName='MergeBlocks1', Input=tmp_restart_0000000)
	
	# show data in view
	mergeBlocks1Display = Show(mergeBlocks1, renderView1, 'UnstructuredGridRepresentation')
	
	# get color transfer function/color map for 'data'
	dataLUT = GetColorTransferFunction('data')
	
	# get opacity transfer function/opacity map for 'data'
	dataPWF = GetOpacityTransferFunction('data')
	
	# trace defaults for the display properties.
	mergeBlocks1Display.Representation = 'Surface'
	mergeBlocks1Display.ColorArrayName = ['CELLS', 'data']
	mergeBlocks1Display.LookupTable = dataLUT
	mergeBlocks1Display.SelectTCoordArray = 'None'
	mergeBlocks1Display.SelectNormalArray = 'None'
	mergeBlocks1Display.SelectTangentArray = 'None'
	mergeBlocks1Display.OSPRayScaleFunction = 'PiecewiseFunction'
	mergeBlocks1Display.SelectOrientationVectors = 'None'
	mergeBlocks1Display.ScaleFactor = 0.8250000000000001
	mergeBlocks1Display.SelectScaleArray = 'data'
	mergeBlocks1Display.GlyphType = 'Arrow'
	mergeBlocks1Display.GlyphTableIndexArray = 'data'
	mergeBlocks1Display.GaussianRadius = 0.04125
	mergeBlocks1Display.SetScaleArray = [None, '']
	mergeBlocks1Display.ScaleTransferFunction = 'PiecewiseFunction'
	mergeBlocks1Display.OpacityArray = [None, '']
	mergeBlocks1Display.OpacityTransferFunction = 'PiecewiseFunction'
	mergeBlocks1Display.DataAxesGrid = 'GridAxesRepresentation'
	mergeBlocks1Display.PolarAxes = 'PolarAxesRepresentation'
	mergeBlocks1Display.ScalarOpacityFunction = dataPWF
	mergeBlocks1Display.ScalarOpacityUnitDistance = 0.03673940176856459
	mergeBlocks1Display.OpacityArrayName = ['CELLS', 'data']
	
	# hide data in view
	Hide(tmp_restart_0000000, renderView1)
	
	# show color bar/color legend
	mergeBlocks1Display.SetScalarBarVisibility(renderView1, True)
	
	# update the view to ensure updated data information
	renderView1.Update()
	
	# create a new 'Resample To Image'
	resampleToImage1 = ResampleToImage(registrationName='ResampleToImage1', Input=mergeBlocks1)
	resampleToImage1.SamplingBounds = [-0.125, 8.125, -0.125, 4.125, -0.125, 4.125]
	
	# Properties modified on resampleToImage1
	#resampleToImage1.SamplingDimensions = [1024, 512, 512]
	resampleToImage1.SamplingDimensions = [2048, 1024, 1024]
	#resampleToImage1.SamplingDimensions = [4096, 2048, 2048]
	
	# show data in view
	resampleToImage1Display = Show(resampleToImage1, renderView1, 'UniformGridRepresentation')
	
	# trace defaults for the display properties.
	resampleToImage1Display.Representation = 'Outline'
	resampleToImage1Display.ColorArrayName = [None, '']
	resampleToImage1Display.SelectTCoordArray = 'None'
	resampleToImage1Display.SelectNormalArray = 'None'
	resampleToImage1Display.SelectTangentArray = 'None'
	resampleToImage1Display.OSPRayScaleArray = 'data'
	resampleToImage1Display.OSPRayScaleFunction = 'PiecewiseFunction'
	resampleToImage1Display.SelectOrientationVectors = 'None'
	resampleToImage1Display.ScaleFactor = 0.8250000000000001
	resampleToImage1Display.SelectScaleArray = 'None'
	resampleToImage1Display.GlyphType = 'Arrow'
	resampleToImage1Display.GlyphTableIndexArray = 'None'
	resampleToImage1Display.GaussianRadius = 0.04125
	resampleToImage1Display.SetScaleArray = ['POINTS', 'data']
	resampleToImage1Display.ScaleTransferFunction = 'PiecewiseFunction'
	resampleToImage1Display.OpacityArray = ['POINTS', 'data']
	resampleToImage1Display.OpacityTransferFunction = 'PiecewiseFunction'
	resampleToImage1Display.DataAxesGrid = 'GridAxesRepresentation'
	resampleToImage1Display.PolarAxes = 'PolarAxesRepresentation'
	resampleToImage1Display.ScalarOpacityUnitDistance = 0.12825654460710195
	resampleToImage1Display.OpacityArrayName = ['POINTS', 'data']
	resampleToImage1Display.SliceFunction = 'Plane'
	resampleToImage1Display.Slice = 31
	
	# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
	resampleToImage1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]
	
	# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
	resampleToImage1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]
	
	# init the 'Plane' selected for 'SliceFunction'
	resampleToImage1Display.SliceFunction.Origin = [4.0, 2.0, 2.0]
	
	# hide data in view
	Hide(mergeBlocks1, renderView1)
	
	# update the view to ensure updated data information
	renderView1.Update()
	
	# set scalar coloring
	ColorBy(resampleToImage1Display, ('POINTS', 'data'))
	
	# rescale color and/or opacity maps used to include current data range
	resampleToImage1Display.RescaleTransferFunctionToDataRange(True, True)
	
	# change representation type
	resampleToImage1Display.SetRepresentationType('Volume')
	
	# Properties modified on resampleToImage1Display
	resampleToImage1Display.ScalarOpacityUnitDistance = 0.005
	
	# Properties modified on renderView1
	renderView1.UseGradientBackground = 1
	
	# Properties modified on renderView1
	renderView1.Background = [0.11140611886778058, 0.2490424963759823, 0.505500877393759]
	
	# Properties modified on renderView1
	renderView1.Background2 = [0.12607003891050583, 0.38460364690623333, 0.7286182955672541]
	
	# set active source
	SetActiveSource(chi_restart_0000000)
	
	# toggle 3D widget visibility (only when running from the GUI)
	Show3DWidgets(proxy=chi_restart_0000000Display.SliceFunction)
	
	# toggle 3D widget visibility (only when running from the GUI)
	Show3DWidgets(proxy=chi_restart_0000000Display)
	
	# toggle 3D widget visibility (only when running from the GUI)
	Hide3DWidgets(proxy=chi_restart_0000000Display.SliceFunction)
	
	# toggle 3D widget visibility (only when running from the GUI)
	Hide3DWidgets(proxy=chi_restart_0000000Display)
	
	# create a new 'Cell Data to Point Data'
	cellDatatoPointData1 = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=chi_restart_0000000)
	cellDatatoPointData1.CellDataArraytoprocess = ['data']
	
	# show data in view
	cellDatatoPointData1Display = Show(cellDatatoPointData1, renderView1, 'UniformGridRepresentation')
	
	# trace defaults for the display properties.
	cellDatatoPointData1Display.Representation = 'Outline'
	cellDatatoPointData1Display.ColorArrayName = ['POINTS', '']
	cellDatatoPointData1Display.SelectTCoordArray = 'None'
	cellDatatoPointData1Display.SelectNormalArray = 'None'
	cellDatatoPointData1Display.SelectTangentArray = 'None'
	cellDatatoPointData1Display.OSPRayScaleArray = 'data'
	cellDatatoPointData1Display.OSPRayScaleFunction = 'PiecewiseFunction'
	cellDatatoPointData1Display.SelectOrientationVectors = 'None'
	cellDatatoPointData1Display.ScaleFactor = 0.8250000000000001
	cellDatatoPointData1Display.SelectScaleArray = 'data'
	cellDatatoPointData1Display.GlyphType = 'Arrow'
	cellDatatoPointData1Display.GlyphTableIndexArray = 'data'
	cellDatatoPointData1Display.GaussianRadius = 0.04125
	cellDatatoPointData1Display.SetScaleArray = ['POINTS', 'data']
	cellDatatoPointData1Display.ScaleTransferFunction = 'PiecewiseFunction'
	cellDatatoPointData1Display.OpacityArray = ['POINTS', 'data']
	cellDatatoPointData1Display.OpacityTransferFunction = 'PiecewiseFunction'
	cellDatatoPointData1Display.DataAxesGrid = 'GridAxesRepresentation'
	cellDatatoPointData1Display.PolarAxes = 'PolarAxesRepresentation'
	cellDatatoPointData1Display.ScalarOpacityUnitDistance = 0.03673940176856459
	cellDatatoPointData1Display.OpacityArrayName = ['POINTS', 'data']
	cellDatatoPointData1Display.IsosurfaceValues = [0.38854074478149414]
	cellDatatoPointData1Display.SliceFunction = 'Plane'
	cellDatatoPointData1Display.Slice = 34
	
	# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
	cellDatatoPointData1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.7770814895629883, 1.0, 0.5, 0.0]
	
	# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
	cellDatatoPointData1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.7770814895629883, 1.0, 0.5, 0.0]
	
	# init the 'Plane' selected for 'SliceFunction'
	cellDatatoPointData1Display.SliceFunction.Origin = [4.0, 2.0, 2.0]
	
	# hide data in view
	Hide(chi_restart_0000000, renderView1)
	
	# update the view to ensure updated data information
	renderView1.Update()
	
	# set scalar coloring
	ColorBy(cellDatatoPointData1Display, ('FIELD', 'vtkBlockColors'))
	
	# show color bar/color legend
	cellDatatoPointData1Display.SetScalarBarVisibility(renderView1, True)
	
	# create a new 'Contour'
	contour1 = Contour(registrationName='Contour1', Input=cellDatatoPointData1)
	contour1.ContourBy = ['POINTS', 'data']
	contour1.Isosurfaces = [0.38854074478149414]
	contour1.PointMergeMethod = 'Uniform Binning'
	
	# Properties modified on contour1
	contour1.ComputeNormals = 0
	contour1.Isosurfaces = [0.5]
	
	# show data in view
	contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')
	
	# trace defaults for the display properties.
	contour1Display.Representation = 'Surface'
	contour1Display.ColorArrayName = ['POINTS', 'data']
	contour1Display.LookupTable = dataLUT
	contour1Display.SelectTCoordArray = 'None'
	contour1Display.SelectNormalArray = 'None'
	contour1Display.SelectTangentArray = 'None'
	contour1Display.OSPRayScaleArray = 'data'
	contour1Display.OSPRayScaleFunction = 'PiecewiseFunction'
	contour1Display.SelectOrientationVectors = 'None'
	contour1Display.ScaleFactor = 0.4320478677749634
	contour1Display.SelectScaleArray = 'data'
	contour1Display.GlyphType = 'Arrow'
	contour1Display.GlyphTableIndexArray = 'data'
	contour1Display.GaussianRadius = 0.02160239338874817
	contour1Display.SetScaleArray = ['POINTS', 'data']
	contour1Display.ScaleTransferFunction = 'PiecewiseFunction'
	contour1Display.OpacityArray = ['POINTS', 'data']
	contour1Display.OpacityTransferFunction = 'PiecewiseFunction'
	contour1Display.DataAxesGrid = 'GridAxesRepresentation'
	contour1Display.PolarAxes = 'PolarAxesRepresentation'
	
	# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
	contour1Display.ScaleTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]
	
	# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
	contour1Display.OpacityTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]
	
	# show color bar/color legend
	contour1Display.SetScalarBarVisibility(renderView1, True)
	
	# update the view to ensure updated data information
	renderView1.Update()
	
	# hide data in view
	Hide(cellDatatoPointData1, renderView1)
	
	# turn off scalar coloring
	ColorBy(contour1Display, None)
	
	# Hide the scalar bar for this color map if no visible data is colored by it.
	HideScalarBarIfNotNeeded(dataLUT, renderView1)
	
	# Properties modified on contour1Display
	contour1Display.Luminosity = 50.0
	
	# Properties modified on contour1Display
	contour1Display.Ambient = 0.01
	
	# Properties modified on contour1Display
	contour1Display.Ambient = 0.5
	
	# set active source
	SetActiveSource(resampleToImage1)
	
	# toggle 3D widget visibility (only when running from the GUI)
	Show3DWidgets(proxy=resampleToImage1Display.SliceFunction)
	
	# toggle 3D widget visibility (only when running from the GUI)
	Show3DWidgets(proxy=resampleToImage1Display)
	
	# toggle 3D widget visibility (only when running from the GUI)
	Hide3DWidgets(proxy=resampleToImage1Display.SliceFunction)
	
	# toggle 3D widget visibility (only when running from the GUI)
	Hide3DWidgets(proxy=resampleToImage1Display)
	
	# hide color bar/color legend
	resampleToImage1Display.SetScalarBarVisibility(renderView1, False)
	
	# Rescale transfer function
	dataLUT.RescaleTransferFunction(0.0, 20.0)
	
	# Rescale transfer function
	dataPWF.RescaleTransferFunction(0.0, 20.0)
	
	# Hide orientation axes
	renderView1.OrientationAxesVisibility = 0
	
		
	#######################
	## CAMERA PATH SETUP ##
	#######################
	# get camera animation track for the view
	cameraAnimationCue1 = GetCameraTrack(view=renderView1)
	
	# create keyframes for this animation track
	
	# create a key frame
	keyFrame10311 = CameraKeyFrame()
	keyFrame10311.Position = [4.0, 2.0, 21.718849954132278]
	keyFrame10311.FocalPoint = [4.0, 2.0, 2.0]
	keyFrame10311.ParallelScale = 5.103613915648401
	keyFrame10311.PositionPathPoints = [4.146176636219025, 1.9862573146820068, 12.916550785303116, 10.541028804162229, 1.9862573146820068, 10.838737361440666,10.48798044014648, 2.2087600053653893, 3.961049470245844,14.15973158458563, 5.168026546542442, -2.1808140561915965,6.601792157112349, 2.431380404894723, -1.9574163442868209,2.2395846140434146, 2.3558249238249287, -2.278269602990524,-11.534015651080466, 2.423900656195463, -2.6767652702614035,8.830342197217833, 2.052128535275979, 4.388868380491843,-11.186944783070556, 2.6254061577221366, 8.279951891541184,-4.474415088817864, 1.871503473225299, 14.712786755304561]
	keyFrame10311.FocalPathPoints = [4.146176636219025, 1.9862573146820068, 2.036978453397751]
	keyFrame10311.ClosedPositionPath = 1
	
	# create a key frame
	keyFrame10312 = CameraKeyFrame()
	keyFrame10312.KeyTime = 1.0
	keyFrame10312.Position = [4.0, 2.0, 21.718849954132278]
	keyFrame10312.FocalPoint = [4.0, 2.0, 2.0]
	keyFrame10312.ParallelScale = 5.103613915648401
	
	# initialize the animation track
	cameraAnimationCue1.Mode = 'Path-based'
	cameraAnimationCue1.KeyFrames = [keyFrame10311, keyFrame10312]
	
	# get animation scene
	animationScene1 = GetAnimationScene()
	
	
	# Properties modified on animationScene1
	animationScene1.PlayMode = 'Snap To TimeSteps'
	#start_a = 1
	#end_a = 1
	fr = 10
	animationScene1.PlayMode = 'Sequence'
	animationScene1.NumberOfFrames = fr*max_frame
	start_a = fr* (case-1)
	end_a   = fr*(case+1-1)
	
	# get the time-keeper
	timeKeeper1 = GetTimeKeeper()
	
	# get layout
	layout1 = GetLayout()
	
	# layout/tab size in pixels
	layout1.SetSize(989, 778)
	
	# current camera placement for renderView1
	renderView1.CameraPosition = [24.106223426078866, 16.169213836950817, 46.842473608370256]
	renderView1.CameraFocalPoint = [4.0000000000000036, 1.9999999999999918, 1.9999999999999971]
	renderView1.CameraViewUp = [0.11021261024760844, 0.932459572248042, -0.34405279633423597]
	renderView1.CameraParallelScale = 5.103613915648401
	
	# save animation
	SaveAnimation(file_name, renderView1, ImageResolution=[1000, 800], FrameWindow=[start_a, end_a], SuffixFormat='.%06d')
	


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--case', required=True, type=int)
	parser.add_argument('--path', required=True, type=str)
	parser.add_argument('--name', required=True, type=str)
	args = vars(parser.parse_args())
	case = args['case']
	path = args['path']
	name = "./" + args['name'] + ".png"
	
	all_chi_files = glob.glob(path+"chi_restart*.xmf")
	all_tmp_files = glob.glob(path+"tmp_restart*.xmf")
	all_chi_files.sort()
	all_tmp_files.sort()
	
	max_index = len(all_chi_files)

	chi_files = []
	tmp_files = []
	indices = []
	#indices.append(0)
	indices.append(case)
	#indices.append(max_index-1)
	chi_files = []
	tmp_files = []
	for i in indices:
		chi_files.append(all_chi_files[i])
		tmp_files.append(all_tmp_files[i])
    
	ParaviewRendering(chi_files, tmp_files, name, max_index, case)

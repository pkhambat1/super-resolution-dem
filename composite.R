library(raster)
library(rgdal)
library(rgeos)

workingdir<-"H:/Shared drives/BU_DEMSuperRes_NW/Data_sets/30_12_2_1_2m_v3.0.tar"

setwd(workingdir)

DEM2m<-raster("Clipped_dem.tif")
GRID1km<-crop(shapefile("Grids.shp"),extent(DEM2m))
individualgrid<-SpatialPolygons(GRID1km@polygons)

outdir<-"H:/Shared drives/BU_DEMSuperRes_NW/Data_sets/30_12_2_1_2m_v3.0.tar/CLIPPEDDEM_HILLSHADE_SLOPE/"

for (i in 2:length(individualgrid)){
  cropdem<-crop(DEM2m,individualgrid[i])
  cropslope<-terrain(cropdem,'slope')
  cropaspect<-terrain(cropdem,'aspect')
  crophillshade<-hillShade(cropslope,cropaspect,angle = 45,direction = 0)
  
  stackall=stack(cropdem,cropslope,crophillshade)
  filename1<-paste0(outdir,"DEM2m_slope_shade_",i,".tif")
  writeRaster(stackall,filename1)
}


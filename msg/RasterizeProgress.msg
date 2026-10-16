# Progress of a map_rasterizer/rasterize request, published by map_rasterizer_server
# on map_rasterizer/progress while it renders.

string basename           # the request's basename, to match progress to a request

uint8 PHASE_LOADING=0     # reading .posegraph + .data; 0 of 1 at start, 1 of 1 when done
uint8 PHASE_RASTERING=1   # done/total are scans
uint8 PHASE_WRITING=2     # done/total are image rows
uint8 phase

uint32 done
uint32 total

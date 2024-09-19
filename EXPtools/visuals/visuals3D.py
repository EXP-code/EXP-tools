
import numpy as np
import k3d

def field3Dcontour(field, contour_range, contour_name, size, **kwargs):
    """
    
    field: numpy.ndarray
        shape(t, N, N, N), where t is the time axis, N are the spatial axis.
    
    contour_ranges: list
        values in percentage of the contour levels. e.g [0.5, 0.75] means
        contours at the 50% and 75% level.  
    
    volume_names 
    
    kwargs:
        color map
    
    """
    
    field_max = np.max(np.abs(field))
    #field_min = np.min(field)


    volume =  k3d.volume(field.astype(np.float32), 
                        alpha_coef=1.0,
                        color_range=contour_range,  
                        color_map=(np.array(k3d.colormaps.paraview_color_maps.Cool_to_Warm_Extended).reshape(-1,4) 
                   * np.array([1,1.0,1.0,1.0])).astype(np.float32),
                        name=contour_name, compression_level=7)

    # Where this values come from? 
    volume.opacity_function  = [0.        , 0.        , 0.21327923, 0.98025   , 0.32439035,
           0.        , 0.5       , 0.        , 0.67560965, 0.        ,
           0.74537706, 0.9915    , 1.        , 0.        ]


    volume.transform.bounds = [-size[0], size[0],
                               -size[1], size[1],
                               -size[2], size[2]]
    
    if 'alpha' in kwargs.keys():
        volume.alpha_coef = kwargs['alpha']
    
    return volume

def orbit3D(orbit, orbit_name, **kwargs):
    """
    # TODO: Add time-dependent 
    
    orbit: numpy.array
        (3, N) shape 
    
    orbit_name 
    
    color 
    
    kwargs:
        color map
    
    """
    
    orbit_trayectory = k3d.line(orbit.astype(np.float32),
                                width=1,
                                color_range=[0, 0.1], color=0x3e3a3a, name=orbit_name)


    
    if 'alpha' in kwargs.keys():
        orbit_trayectory.alpha_coef = kwargs['alpha']
    
    return orbit_trayectory



def field3Drender(volumes, contour_ranges, size, contour_names=['Inner', 'Outter'],
        contour_alphas=[8,1], **kwargs):
    """
    volumes: numpy.ndarray
        shape: t, gridx, gridy, gridz
        
    orbit

    """
    render = k3d.plot(height=800)

    for vol in range(len(contour_ranges)):
        v = field3Dcontour(volumes[0], 
                           contour_range=contour_ranges[vol], 
                           contour_name=contour_names[vol], 
                           alpha=contour_alphas[vol], size=size)
        render += v
        v_t = {}
        for t in range(volumes.shape[0]):
            v_t[str(float(t))] = volumes[t].astype(np.float32)
        v.volume = v_t
    
    # Implement 3d orbit
    if 'orbits' in kwargs.keys():
        
        for orb in range(len(kwargs['orbits_names'])):
            o = orbit3D(kwargs['orbits'][orb], 
                        orbit_name=kwargs['orbits_names'][orb], width=10)
            render += o

    render.grid = (-size[0], -size[1], -size[2], size[0], size[1], size[2])
    return render

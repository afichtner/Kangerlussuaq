import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times"
plt.rcParams.update({'font.size': 30})
plt.rcParams['xtick.major.pad']='12'
plt.rcParams['ytick.major.pad']='12'

def plot(x0,y0,x,y,xs,ys,filename=None):
    """
    Plot geometry of the cable and the sources, possibly also the obstacles.

    x0, y0: x- and y-coordinate arrays of initial points that the cable must cross
    x, y: x- and y-coordinate arrays of refined points along the cable
    xs, ys: x- and y-coordinates of sources.
    """
    
    plt.figure(figsize=(12,12))

    # Plot obstacles. =========================================================

    xo=np.arange(-1000.0,6000.0,10)
    yo1=900.0*np.ones(np.shape(xo))
    yo2=5000.0*np.ones(np.shape(xo))
    plt.fill_between(xo,yo1,yo2,facecolor=[0.8,0.8,0.8],alpha=0.5)

    xo=np.arange(-1000.0,0.0,10)
    yo1=-2000.0*np.ones(np.shape(xo))
    yo2=910.0*np.ones(np.shape(xo))
    plt.fill_between(xo,yo1,yo2,facecolor=[0.8,0.8,0.8],alpha=0.5)

    xo=np.arange(1500.0,2000.0,10)
    yo1=-1100.0*np.ones(np.shape(xo))
    yo2=-100.0*np.ones(np.shape(xo))
    plt.fill_between(xo,yo1,yo2,facecolor=[0.8,0.8,0.8],alpha=0.5)

    xo=np.arange(500.0,1000.0,10)
    yo1=400.0*np.ones(np.shape(xo))
    yo2=700.0*np.ones(np.shape(xo))
    plt.fill_between(xo,yo1,yo2,facecolor=[0.8,0.8,0.8],alpha=0.5)

    # Plot current interpolation points. ======================================

    plt.plot(x,y,'k',linewidth=2)
    #plt.plot(x,y,'ko')

    # Plot source location. ===================================================

    plt.plot(xs,ys,'r*',markersize=15)

    # Plot initial interpolation points. ======================================

    plt.plot(x0,y0,'bo',markersize=15)
    
    # Embellishments. =========================================================

    plt.xlim(-200.0,3200.0)
    plt.ylim(-1200.0,3200.0)

    plt.xlabel('x [m]',labelpad=10)
    plt.ylabel('y [m]',labelpad=10)
    plt.grid()
    plt.tight_layout()

    if filename: plt.savefig(filename,dpi=200,format='png')

    plt.show()


#==================================================================================================
#==================================================================================================

def obstacles(x,y):
    """
    Function defining the obstacles that a new point along the cable should avoid. 
    The function must return True when a new point falls into an obstacle that is to be avoided.
    """
    if y>900.0:
        return True

    elif x<0.0: 
        return True

    elif x<2000.0 and x>1500.0 and y<-100.0 and y>-1100.0:
        return True

    elif x<1000.0 and x>500.0 and y<700.0 and y>400.0:
        return True

    return False


#==================================================================================================
#==================================================================================================

def check_obstacles(x1,y1,x2,y2,x3,y3,dl):
    """
    Check if a newly created cable segment crosses one of the obstacles. For this, we take the
    coordinates of the previous points, (x1,y1) and (x3,y3), as well as the coordinates of the
    new point inbetween, (x2,y2). Then we compute channel coordinates along this cable stretch.
    This requires knowledge of the channel spacing dl. Then, for each channel coordinate, we
    call the obstacles function, which checks if an individual point falls inside an obstacle.
    """

    xc,yc=make_channel_coordinates(np.array([x1,x2,x3]), np.array([y1,y2,y3]), 3.0*dl)
    in_obstacle=False

    for i in range(len(xc)):
        if obstacles(xc[i],yc[i]): 
            in_obstacle=True
            break

    if in_obstacle:
        return True
    else:
        return False


#==================================================================================================
#==================================================================================================

def insert_points(x_in, y_in, w_in, m, L, dl):
    """
    x_in, y_in: Current x- and y-positions of interpolation points along the cable.
    w_in: Current vector of weights (<=1) for the maximum lengths of the new segments.
    m: Exponent for the calculation of new weights (<=1).
    L: Maximum allowable length of the cable.
    dl: Channel spacing.
    return: New interpolation point coordinates and new weights.
    """

    # Maximum number of random trials to attempt avoidance of an obstacle.
    n_trials=20
    
    # Random drawing of lengths for subdivided segments. ======================

    # Number of incoming points.
    n=len(x_in)
    
    # Make vector of lengths.
    l_in=np.sqrt((x_in[1:n]-x_in[0:n-1])**2 + (y_in[1:n]-y_in[0:n-1])**2)

    # Initiate new array of x- and y-coordinates.
    x=np.zeros(2*n-1)
    y=np.zeros(2*n-1)
    x[0:2*n+1:2]=x_in
    y[0:2*n+1:2]=y_in


    # Randomly draw lengths of new segments. This part implements a specific length distribution function
    # that is uniform between l_min and l_max.
    l_min=l_in/2.0
    l_max=l_in*L/(2.0*np.sum(l_in))
    l=w_in*(l_max-l_min)*np.random.rand(n-1)+l_min

    # Compute new x- and y-coordinates while trying to circumvent obstacles. ==
    for i in range(n-1):
        
        # Components of the perpendicular a vector.
        a=np.sqrt(4.0*l[i]**2-l_in[i]**2)
        ax=a*(y_in[i+1]-y_in[i])/l_in[i]
        ay=-a*(x_in[i+1]-x_in[i])/l_in[i]

        # Randomly choose new x- and y-coordinates.
        for trials in range(n_trials):

            if np.random.rand()>0.5: 
                sign=1.0
            else:
                sign=-1.0

            x[2*i+1]=0.5*(x_in[i]+x_in[i+1])+0.5*sign*ax
            y[2*i+1]=0.5*(y_in[i]+y_in[i+1])+0.5*sign*ay

            # Choose the other option in case point is inside obstacle.
            if check_obstacles(x[2*i],y[2*i],x[2*i+1],y[2*i+1],x[2*i+2],y[2*i+2],dl):
                x[2*i+1]=0.5*(x_in[i]+x_in[i+1])-0.5*sign*ax
                y[2*i+1]=0.5*(y_in[i]+y_in[i+1])-0.5*sign*ay

            # Check if this was actually successful. If so, stop trying.
            if not check_obstacles(x[2*i],y[2*i],x[2*i+1],y[2*i+1],x[2*i+2],y[2*i+2],dl):
                success=True
                break

            # If not, set success to false.
            if check_obstacles(x[2*i],y[2*i],x[2*i+1],y[2*i+1],x[2*i+2],y[2*i+2],dl) and trials==(n_trials-1):
                w_in[i]=w_in[i]**m
                success=False

        # Terminate loop in case one point falls into obstacle.
        if success==False:
            break
        
    # Lengths of the new segments.
    l=np.sqrt((x[1:2*n-1]-x[0:2*n-2])**2 + (y[1:2*n-1]-y[0:2*n-2])**2)

    # New weight vector.
    w=np.zeros(2*n-2)
    w[0:2*n+1:2]=w_in**m
    w[1:2*n:2]=w_in**m

    # Return.
    return success,x, y, w


#==================================================================================================
#==================================================================================================

def make_channel_coordinates(x, y, dl):
    """
    Compute x- and y-coordinates of channels along the cable.
    x, y: current interpolation points along the cable.
    dl: channel spacing.
    Returns channel locations xc and yc.
    """

    # Unit vector pointing from one point to the next.
    n=len(x)
    l=np.sqrt((x[1:n]-x[0:n-1])**2 + (y[1:n]-y[0:n-1])**2)
    ex=(x[1:n]-x[0:n-1])/l
    ey=(y[1:n]-y[0:n-1])/l

    # Initialise channel locations.
    xc=[]
    yc=[]

    # March through current interpolation points and place channels inbetween.
    for i in range(n-1):

        # Vector of step lengths between channels.
        s=np.arange(0.0,l[i],dl)
        # Append new channel locations.
        xc=np.append(xc,x[i]+s*ex[i])
        yc=np.append(yc,y[i]+s*ey[i])

    # Return.
    return xc, yc


#==================================================================================================
#==================================================================================================

def surface(x, y):
    """
    Function that defines topographic surface. Takes x- and y-coordinates in metres and returns
    topography in metres.
    """
    
    return 200.0+150.0*np.sin(np.sqrt(((x+700.0)/400.0)**2 + ((y+1000.0)/400.0)** 2))


#==================================================================================================
#==================================================================================================

def axisEqual3D(ax):
    
    extents=np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz=extents[:,1]-extents[:,0]
    centers=np.mean(extents, axis=1)
    maxsize=max(abs(sz))
    r=maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        if dim=='z':
            getattr(ax, 'set_{}lim'.format(dim))(ctr - r/4.0, ctr + r/2.0)
        else:
            getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)



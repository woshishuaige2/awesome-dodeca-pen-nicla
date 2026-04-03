import itertools as it
import numpy as np
from numpy import linalg as LA

# default is 1 for the inner cube of length 2, scale it to make equal to your pentagon side 
# this scale equal a dodecaheda of side d=13.2mm
# scale_fac = d*cos(36)
scale_fac = 10.679
# can be enlarged to a maximum 12.7
marker_size_in_mm = 12.5
#vertex coordinates
v=[p for p in it.product((-1,1),(-1,1),(-1,1))]
g=.5+.5*5**.5
v.extend([p for p in it.product((0,),(-1/g,1/g),(-g,g))])
v.extend([p for p in it.product((-1/g,1/g),(-g,g),(0,))])
v.extend([p for p in it.product((-g,g),(0,),(-1/g,1/g))])
v=np.array(v)
#20 vertices indices
g=[[12,14,5,9,1],\
    [12,1,17,16,0],\
    [12,0,8,4,14],\
    [4,18,19,5,14],\
    [4,8,10,6,18],\
    [5,19,7,11,9],\
    [7,15,13,3,11],\
    [7,19,18,6,15],\
    [6,10,2,13,15],\
    [13,2,16,17,3],\
    [3,17,1,9,11],\
    [16,2,10,8,0]]

Scl_mat = [[scale_fac,0, 0],[0, scale_fac, 0],[0, 0, scale_fac]] ;
for i in range(20):
    v[i,:] = np.matmul(Scl_mat,v[i,:])

def get_face_frame(vertices):
    "input is a 5 x n array of vertex points"
    x_ax = vertices[0,:] - vertices[2,:] 
    x_ax = x_ax/LA.norm(x_ax,2)
    z_ax = np.cross(vertices[0,:] - vertices[1,:], vertices[0,:] - vertices[3,:])
    z_ax = z_ax/LA.norm(z_ax,2)
    y_ax = np.cross (z_ax, x_ax)
    
    x_ax = np.reshape(x_ax,(3,1))
    y_ax = np.reshape(y_ax,(3,1))
    z_ax = np.reshape(z_ax,(3,1))
    
    # print(x_ax.shape)
    
    r_mat= np.hstack((x_ax,y_ax,z_ax))
    
    face_center = np.array([np.mean(vertices[:,0]), np.mean(vertices[:,1]), np.mean(vertices[:,2])])
    face_center = np.reshape(face_center,(3,1))
    
    return face_center, r_mat

def tf_mat_dodeca_pen(face_id):
	'''
	Function that looks at the dodecahedron geometry to get the rotation matrix and translation
	TODO: when in class have to pass the dodecahedron geometry to this as a variable
	Inputs: face_id: the face for which the transformation matrices is quer=ries (int)
	Outputs: T_mat_cent_face = transformation matrix from center of the dodecahedron to a face
	T_mat_face_cent = transformation matrix from face (with given face id) to the dodecahedron center
	
	'''
	T_cent_face_curr = T_cent_face[face_id,:,:]
	_R_cent_face_curr = R_cent_face[face_id,:,:]
	T_mat_cent_face = np.vstack((np.hstack((_R_cent_face_curr,T_cent_face_curr)),np.array([0,0,0,1])))
	T_mat_face_cent = np.vstack((np.hstack((_R_cent_face_curr.T,-_R_cent_face_curr.T.dot(T_cent_face_curr))),np.array([0,0,0,1])))
	return T_mat_cent_face,T_mat_face_cent

def corners_3d(tf_mat,m_s):
	'''
	Function to give coordinates of the marker corners and transform them using a given transformation matrix
	Inputs:
	tf_mat = transformation matrix between face frames and dodeca frames
	m_s = marker size-- edge lenght in mm
	Outputs:
	corn_pgn_f = corners in camara frame
	'''
	corn_1 = np.array([-m_s/2.0,  m_s/2.0, 0, 1])
	corn_2 = np.array([ m_s/2.0,  m_s/2.0, 0, 1])
	corn_3 = np.array([ m_s/2.0, -m_s/2.0, 0, 1])
	corn_4 = np.array([-m_s/2.0, -m_s/2.0, 0, 1])
	corn_mf = np.vstack((corn_1,corn_2,corn_3,corn_4))
	corn_pgn_f = tf_mat.dot(corn_mf.T)
	return corn_pgn_f

R_cent_face = np.zeros((12,3,3))
T_cent_face = np.zeros((12,3,1))

for f in range(12):
    y = v[g[f],:]
    (T_cent_face[f,:,:], R_cent_face[f,:,:] ) = get_face_frame(y)

# print(T_cent_face)
# print(R_cent_face)

corners_in_cart_sp = np.zeros((12,4,3))

for ii in range(12):
    Tf_cent_face,Tf_face_cent = tf_mat_dodeca_pen(ii)
    corners_in_cart_sp[ii,:,:] = corners_3d(Tf_cent_face, marker_size_in_mm).T[:,0:3]

# print(corners_in_cart_sp)


np.save('12 marker corner coordinates.npy',corners_in_cart_sp)
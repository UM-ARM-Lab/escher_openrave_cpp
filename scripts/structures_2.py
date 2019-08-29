import numpy as np
import math

from transformation_conversion import *

surface_slice_resolution = 0.001

class structure:
	def __init__(self,identity,geometry,kinbody=None):
		self.id = identity
		self.geometry = geometry
		self.kinbody = kinbody

class trimesh_surface(structure):
	def __init__(self,identity,plane_parameters,vertices,boundaries,mesh=None,trimesh_vertices=None,trimesh_indices=None):

		coefficient_norm = math.sqrt(plane_parameters[0]**2 + plane_parameters[1]**2 + plane_parameters[2]**2)

		# nx * x + ny * y + nz * z + c = 0
		self.nx = plane_parameters[0]/coefficient_norm
		self.ny = plane_parameters[1]/coefficient_norm
		self.nz = plane_parameters[2]/coefficient_norm
		self.c = plane_parameters[3]/coefficient_norm

		self.vertices = vertices
		self.boundaries = boundaries

		self.xo = 0.0; self.yo = 0.0; self.zo = 0.0

		self.circumscribed_radius = 0
		for i in range(len(self.vertices)-1):
			for j in range(i+1,len(self.vertices)):
				dist = math.sqrt((self.vertices[i][0]-self.vertices[j][0])**2 + (self.vertices[i][1]-self.vertices[j][1])**2 + (self.vertices[i][2]-self.vertices[j][2])**2)
				if(dist/2.0 > self.circumscribed_radius):
					self.circumscribed_radius = dist/2.0
					self.xo = (self.vertices[i][0]+self.vertices[j][0])/2.0
					self.yo = (self.vertices[i][1]+self.vertices[j][1])/2.0
					self.zo = (self.vertices[i][2]+self.vertices[j][2])/2.0

		max_edge_length = 0
		edge_vector = [0,0,0]
		for i in range(len(self.vertices)):
			if(i != 0):
				edge_length = np.linalg.norm(np.array(self.vertices[i]) - np.array(self.vertices[i-1]))

				if(edge_length > max_edge_length):
					max_edge_length = edge_length
					for j in range(3):
						edge_vector[j] = self.vertices[i][j]-self.vertices[i-1][j]

		xv = np.array([[edge_vector[0]],[edge_vector[1]],[edge_vector[2]]])
		xv = xv / np.linalg.norm(xv)
		yv = np.cross(self.get_normal().T,xv.T).T
		zv = self.get_normal()

		rotation_matrix = np.hstack((xv,yv,zv))

		self.transform = np.eye(4,dtype=float)
		self.inverse_transform = np.eye(4,dtype=float)

		self.transform[0:3,0:3] = rotation_matrix
		self.transform[0:3,3:4] = self.get_center()

		self.inverse_transform[0:3,0:3] = rotation_matrix.T
		self.inverse_transform[0:3,3:4] = np.dot(-rotation_matrix.T,self.get_center())

		self.max_proj_x = -9999.0
		self.min_proj_x = 9999.0
		self.max_proj_y = -9999.0
		self.min_proj_y = 9999.0

		self.projected_vertices = []
		for vertex in self.vertices:
			proj_vertex = self.projection_plane_frame(np.array([[vertex[0]],[vertex[1]],[vertex[2]]]))
			self.projected_vertices.append((proj_vertex[0,0],proj_vertex[1,0]))

			if(proj_vertex[0,0] < self.min_proj_x):
				self.min_proj_x = proj_vertex[0,0]

			if(proj_vertex[0,0] > self.max_proj_x):
				self.max_proj_x = proj_vertex[0,0]

			if(proj_vertex[1,0] < self.min_proj_y):
				self.min_proj_y = proj_vertex[1,0]

			if(proj_vertex[1,0] > self.max_proj_y):
				self.max_proj_y = proj_vertex[1,0]

		self.projected_vertices.append(self.projected_vertices[0])

		self.approx_boundary = {}
		self.approx_area = 0.0
		self.strip_decomposition = []
		self.calculate_approximate_boundary()

		if(abs(self.nz) > 0.85):
			self.type = 'ground'
		else:
			self.type = 'others'

		# for reconstructing surface kinbody
		self.trimesh_vertices = trimesh_vertices
		self.trimesh_indices = trimesh_indices

		structure.__init__(self,identity,'trimesh',mesh)


	def transform_data(self,transform,mesh):
		normal_vector = np.vstack((self.get_normal(),np.array([0])))

		transformed_normal_vector = np.dot(transform,normal_vector)
		self.nx = transformed_normal_vector[0,0]
		self.ny = transformed_normal_vector[1,0]
		self.nz = transformed_normal_vector[2,0]

		new_vertices = []

		for vertex in self.vertices:
			vertex_position = np.vstack((np.atleast_2d(np.array(vertex)).T,np.array([1])))
			transformed_vertex_position = np.dot(transform,vertex_position)
			new_vertices.append((transformed_vertex_position[0,0],transformed_vertex_position[1,0],transformed_vertex_position[2,0]))

		self.vertices = new_vertices

		self.c = -self.nx * self.vertices[0][0] - self.ny * self.vertices[0][1] - self.nz * self.vertices[0][2]

		transformed_center = np.dot(transform,np.vstack((self.get_center(),np.array([1]))))
		self.xo = transformed_center[0,0]; self.yo = transformed_center[1,0]; self.zo = transformed_center[2,0]

		self.transform = np.dot(transform, self.transform)
		self.inverse_transform = inverse_SE3(self.transform)
		

	def calculate_approximate_boundary(self):

		for x in range(int(math.floor((self.max_proj_x-self.min_proj_x)/surface_slice_resolution))+1):
			self.approx_boundary[x] = []


		for boundary in self.boundaries:
			x1 = self.projected_vertices[boundary[0]][0]
			y1 = self.projected_vertices[boundary[0]][1]
			x2 = self.projected_vertices[boundary[1]][0]
			y2 = self.projected_vertices[boundary[1]][1]

			if(x1 > x2):
				xl = x1; yl = y1
				xs = x2; ys = y2
			elif(x1 < x2):
				xl = x2; yl = y2
				xs = x1; ys = y1
			else:
				continue

			x_limit1 = int(math.ceil((xs-self.min_proj_x)/surface_slice_resolution))
			x_limit2 = int(math.floor((xl-self.min_proj_x)/surface_slice_resolution))

			for i in range(x_limit1,x_limit2+1):
				x = i*surface_slice_resolution + self.min_proj_x
				y = ((x-xs) * yl + (xl-x) * ys) / (xl-xs)
				self.approx_boundary[i].append(y)

		for key,y_boundaries in self.approx_boundary.iteritems():
			y_boundaries.sort()

			for i,y_boundary in enumerate(y_boundaries):
				if(i % 2 == 1):
					self.approx_area = self.approx_area + (y_boundaries[i] - y_boundaries[i-1]) * surface_slice_resolution
					self.strip_decomposition.append((self.approx_area, key*surface_slice_resolution + self.min_proj_x, y_boundaries[i-1], y_boundaries[i]))

	def get_normal(self):
		return np.array([[self.nx],[self.ny],[self.nz]])

	def get_center(self):
		return np.array([[self.xo],[self.yo],[self.zo]])

	def getTransform(self):
		return self.transform

	def getInverseTransform(self):
		return self.inverse_transform

	def get_global_position(self,proj_p):
		homogeneous_proj_p = np.vstack((proj_p,np.array([[0],[1]])))
		p = np.dot(self.transform,homogeneous_proj_p)

		return p[0:3,0:1]

	# get the projection transformation given an origin and a ray casting from the origin
	def projection(self,robot_obj,origin,ray,roll,end_effector_type,valid_contact=False):
		translation = self.projection_global_frame(origin,ray)

		if(translation is None):
			return None

		if(np.linalg.norm(translation - self.get_center()) > self.circumscribed_radius):
			return None

		if(end_effector_type == 'foot'):
			cz = self.get_normal()
			cx = np.array([[math.cos(roll*DEG2RAD)],[math.sin(roll*DEG2RAD)],[0]])
			cy = np.cross(cz.T,cx.T).T
			cy = cy / np.linalg.norm(cy)
			cx = np.cross(cy.T,cz.T).T

			contact_type = 'foot'
		elif(end_effector_type == 'left_hand'):
			if(abs(np.dot(self.get_normal().T,np.array([0,0,1]))) < 0.9999):
				cx = -self.get_normal()
				cy = np.array([[math.sin(roll*DEG2RAD)],[0],[math.cos(roll*DEG2RAD)]])
				cy = (self.projection_global_frame(translation+cy) - translation)
				cy = cy / np.linalg.norm(cy)
				cz = np.cross(cx.T,cy.T).T
			else:
				cx = -self.get_normal()
				cy = np.array([[math.cos(roll*DEG2RAD)],[math.sin(roll*DEG2RAD)],[0]])
				cy = (self.projection_global_frame(translation+cy) - translation)
				cy = cy / np.linalg.norm(cy)
				cz = np.cross(cx.T,cy.T).T

			contact_type = 'hand'
		elif(end_effector_type == 'right_hand'):
			if(abs(np.dot(self.get_normal().T,np.array([0,0,1]))) < 0.9999):
				cx = -self.get_normal()
				cy = np.array([[-math.sin(roll*DEG2RAD)],[0],[-math.cos(roll*DEG2RAD)]])
				cy = (self.projection_global_frame(translation+cy) - translation)
				cy = cy / np.linalg.norm(cy)
				cz = np.cross(cx.T,cy.T).T
			else:
				cx = -self.get_normal()
				cy = np.array([[-math.cos(roll*DEG2RAD)],[-math.sin(roll*DEG2RAD)],[0]])
				cy = (self.projection_global_frame(translation+cy) - translation)
				cy = cy / np.linalg.norm(cy)
				cz = np.cross(cx.T,cy.T).T

			contact_type = 'hand'

		transform = np.eye(4)
		transform[0:3,0:1] = cx
		transform[0:3,1:2] = cy
		transform[0:3,2:3] = cz
		transform[0:3,3:4] = translation

		valid_contact = self.contact_inside_polygon(robot_obj,transform,contact_type)

		if(valid_contact):
			return transform
		else:
			return None

	def projection_plane_frame(self,p,ray=None):
		if(ray is None):
			p = np.vstack((p,1))
			return np.dot(self.getInverseTransform(),p)[0:2,0:1]
		else:
			n = self.get_normal()

			if(np.dot(n.T,ray)[0] != 0):
				t = (-self.c-np.dot(n.T,p)[0]) / np.dot(n.T,ray)[0]
				proj_p = p + t*ray

				if(t >= 0 and abs(proj_p[0,0]) < 9999.0 and abs(proj_p[1,0]) < 9999.0 and abs(proj_p[2,0]) < 9999.0):
					proj_p = np.vstack((proj_p,1))
					return np.dot(self.getInverseTransform(),proj_p)[0:2,0:1]

		return None

	def projection_global_frame(self,p,ray=None):
		if(ray is None):
			v = p - self.get_center()
			proj_v = v - np.dot(self.get_normal().T,v) * self.get_normal()

			return proj_v + self.get_center()
		else:
			n = self.get_normal().T
			cos = np.dot(n,ray)[0,0]

			if(abs(cos) > 0.001):
				t = (-self.c-np.dot(n,p)[0,0]) / cos

				if(t >= 0):
					proj_p = p + t*ray
					return proj_p

		return None

	# underlying assumption: the polygon is convex
	def contact_inside_polygon(self,robot_obj,tf,contact_type):
		if(contact_type == 'foot'):
			h = robot_obj.foot_h/2.0
			w = robot_obj.foot_w/2.0
			vertices = np.array([[h,h,-h,-h],[w,-w,w,-w],[0,0,0,0],[1,1,1,1]])
		elif(contact_type == 'hand'):
			h = robot_obj.hand_h/2.0
			w = robot_obj.hand_w/2.0
			vertices = np.array([[0,0,0,0],[h,h,-h,-h],[w,-w,w,-w],[1,1,1,1]])

		else:
			print('Error: Unexpected contact contact_type: %s'%contact_type)
			raw_input()
			return False

		transformed_vertices = np.dot(tf,vertices)

		for i in range(transformed_vertices.shape[1]):
			if(not self.inside_polygon(transformed_vertices[0:3,i:i+1])):
				return False

		return True

	def inside_polygon(self,p):
		error_tolerance = 0.005
		x = p[0,0]; y = p[1,0]; z = p[2,0]
		if(abs(self.nx * x + self.ny * y + self.nz * z + self.c) < error_tolerance): # check if the point is in the plane, with an error tolerance
			if(math.sqrt((x-self.xo)**2 + (y-self.yo)**2 + (z-self.zo)**2) < self.circumscribed_radius):
				proj_p = self.projection_plane_frame(p)

				return self.inside_polygon_plane_frame(proj_p)

			else:
				return False

		else:
			return False

	def inside_polygon_plane_frame(self,proj_p):
		px = proj_p[0,0]
		py = proj_p[1,0]

		if(px < self.max_proj_x and px > self.min_proj_x):
			query_x = int(math.floor((px-self.min_proj_x) / surface_slice_resolution))
			y_boundaries = self.approx_boundary[query_x]

			pass_boundary_time = 0
			for i,y_boundary in enumerate(y_boundaries):
				if(py >= y_boundary):
					pass_boundary_time = i+1
				else:
					break

			if(pass_boundary_time % 2 == 1):
				return True
			else:
				return False
		else:
			return False

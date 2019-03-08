import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from random import sample

'''
calculate the probility of each symbol
'''

def infoEntropy():
	"""
	:param bitNum: the bite number of a symbol
	:param N: the number of symbol
	:return:
	"""
	p = [i/1000 for i in range(1000)]
	p=np.array(p)

	#p_syboml = np.zeros((1,16),dtype=float)

	p1 = p * np.power(1-p,3)
	p2 = np.power(p,2) * np.power(1-p, 2)
	p3 = np.power(p, 3) * (1-p)
	p4 = np.power(p, 4)
	p0 = np.power(1-p, 4)

	infoEn = -3 * p1 * np.log2(p1) - 6 * p2 * np.log2(p2) - 5 * p3 * np.log2(p3) \
			- p4 * np.log2(p4) - p0 * np.log2(p0)

	infoTH = np.array([3 for i in range(1000)])

	error = infoTH - infoEn
	for i in range(1000):
		if(abs(error[i])<0.001):
			index = i

	plt.plot(p, infoEn, 'k-', p, infoTH,'r-')
	plt.title("Probility curve with the constrain of information entropy is 3")
	#idx = np.argwhere(np.isclose(infoEn, infoTH, atol=0.001)).reshape(-1)
	plt.plot(p[255], infoEn[255], 'go',p[822],infoTH[822],'bo')
	plt.xlabel("Probility of symbol 0")
	plt.ylabel("Information Entropy")

	plt.savefig("infoEntropy")
	plt.close()

def checkConstarin1(N):
	"""
	find the prob that adhere constrain1
	:return:
	"""
	p_sym = []
	p = [i/1000 for i in range(1000)]
	p = np.array(p,dtype=float)
	#np.random.seed(1)
	#index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
	#id = sample(index,N)
	id = [6, 9, 1, 7, 5, 2, 4, 15, 0, 11, 14, 12]
	print(id,N)

	#for j in range(bitNum):
	#	p_sym.append(p**j * np.power(1-p,bitNum-j))

	# constrain function

	#for j in range(bitNum):
	p1 = p * np.power(1 - p, 3)
	p2 = np.power(p, 2) * np.power(1 - p, 2)
	p3 = np.power(p, 3) * (1 - p)
	p4 = np.power(p, 4)
	p0 = np.power(1 - p, 4)


	p_symbol = np.array([[p1, p1], [p2, p2], [p1, p1], [p0, p0],
						 [p2, p2], [p3, p3], [p2, p2], [p1, p1],
						 [p3, p3], [p4, p4], [p3, p3], [p2, p2],
						 [p2, p2], [p3, p3], [p2, p2], [p3, p3]])

	#  map coordinate to symbol
	'''
	sym_coor = {}

	sym_coor[(1,1)]   = p2
	sym_coor[(1,-1)]  = p3
	sym_coor[(-1,-1)] = p4
	sym_coor[(-1,1)]  = p3
	sym_coor[(1,3)]   = p1
	sym_coor[(3,3)]   = p0
	sym_coor[(3,1)]   = p1
	sym_coor[(3,-1)]  = p2
	sym_coor[(3,-3)]  = p3
	sym_coor[(1,-3)]  = p2
	sym_coor[(-1,-3)] = p3
	sym_coor[(-3,-3)] = p2
	sym_coor[(-3,-1)] = p3
	sym_coor[(-3,1)]  = p2
	sym_coor[(-3,3)]  = p1
	sym_coor[(-1,3)]  = p2
	'''
	s = np.array([[-3, 3], [-1, 3], [1, 3], [3, 3],
				  [-3, 1], [-1, 1], [1, 1], [3, 1],
				  [-3, -1], [-1, -1], [1, -1], [3, -1],
				  [-3, -3], [-1, -3], [1, -3], [3, -3]])  # sort as the same with sym_coor

	avrage_power = np.zeros((2,1000),dtype=float)
	infoEntrop = np.zeros((1,1000),dtype=float)

	for i in range(N):
		avrage_power[0]  += s[id[i]][0] * p_symbol[id[i]][0]
		avrage_power[1]  += s[id[i]][1] * p_symbol[id[i]][0]
		infoEntrop += -p_symbol[id[i]][0] * np.log2(p_symbol[id[i]][0])


	plt.plot(p,avrage_power[0],'b',label="x-axis symbol average power")
	plt.plot(p,avrage_power[1],'r-',label="y-axis symbol average power")
	plt.plot(p,infoEntrop.T,'k',label="information entropy")
	plt.legend()
	plt.title("symbol average power = 0 and information entrop = 3")
	plt.show()
	plt.close()

	return 0

def getInitP_symbol(p):
	"""
	get initial consttelation symbol probility from probility 0
	:param p: probility of 0
	:return:
	"""

	p1 = p * np.power(1 - p, 3)
	p2 = np.power(p, 2) * np.power(1 - p, 2)
	p3 = np.power(p, 3) * (1 - p)
	p4 = np.power(p, 4)
	p0 = np.power(1 - p, 4)

	id = [6, 9, 1, 7, 5, 2, 4, 15, 0, 11, 14, 12]



	p_symbol = np.array([[p1,p1],[p2,p2],[p1,p1],
						 [p2,p2],[p3,p3],[p2,p2],[p1,p1],
						         [p4,p4],        [p2,p2],
						 [p2,p2],        [p2,p2],[p3,p3]])

	#  map coordinate to symbol

	s = np.array([[-3,3],[-1,3],[1,3],
				  [-3,1],[-1,1],[1,1],[3,1],
				         [-1,-1],     [3,-1],
				  [-3,-3],      [1,-3],[3,-3]]) # sort as the same with sym_coor

	'''
	p_symbol = np.array([[p1, p1], [p2, p2], [p1, p1], [p0,p0],
						 [p2, p2], [p3, p3], [p2, p2], [p1, p1],
						 [p3,p3] , [p4, p4],  [p3,p3], [p2, p2],
						 [p2, p2], [p3,p3] ,  [p2, p2], [p3, p3]])

	#  map coordinate to symbol

	s = np.array([[-3, 3], [-1, 3], [1, 3],[3,3],
				  [-3, 1], [-1, 1], [1, 1], [3, 1],
				[-3,-1]	  , [-1, -1],   [1,-1]   ,  [3, -1],
				  [-3, -3],  [-1,-1],       [1, -3], [3, -3]])  # sort as the same with sym_coor
	'''
	'''
	p_symbol = np.array([[1/8,1/8], [1/8,1/8], [1/8,1/8], [1/8,1/8],
						 [1/8,1/8], [1/8,1/8], [1/8,1/8], [1/8,1/8]])

	#  map coordinate to symbol

	s = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1],
				  [0, np.sqrt(3) + 1], [np.sqrt(3) + 1, 0], [0, -np.sqrt(3) - 1], [-np.sqrt(3) - 1, 0]], dtype=float)
	'''
	x = []
	y = []
	for i in range(12):
		x.append(s[i][0])
		y.append(s[i][1])

	plt.scatter(x,y,s=32,c='b')
	plt.savefig("consttelation")
	plt.close()

	constraint1 = p_symbol*s
	res = np.sum(constraint1,axis=0) # check constrain 1

	assert len(res) == 2
	'''
	if(res[0]+res[1]<0.00001):
		print("the initial consttelation adhere the constrain 1 !")
	'''
	return p_symbol, s, res

def getInitP_symbol8():

	p_symbol = np.array([1/8, 1/8, 1/8, 1/8,
	                     1/8, 1/8, 1/8, 1/8])

	#  map coordinate to symbol

	s = np.array([[1,1],[1,-1],[-1,-1],[-1,1],
				  [0,np.sqrt(3)+1],[np.sqrt(3)+1,0],[0,-np.sqrt(3)-1],[-np.sqrt(3)-1,0]],dtype=float)

	return p_symbol,s

def PS_err(s1, s2, p, N0, i1,i2):
	"""
	the prob of decode s2 s1 new position
	:param s1:
	:param s2:
	:param s:
	:param p:
	:param N0:
	:return:
	"""
	x = np.linalg.norm(s1 - s2) / np.sqrt(2 * N0) + \
		(np.sqrt(2*N0) * np.log(p[i2][0] / p[i1][0])) / (2 * np.linalg.norm(s1 - s2))

	res = 0.5 * erfc(x/np.sqrt(2))

	return res


def objectFunc(i1,i2,p,s1,s2,s,N0):
	'''
	during each iteration, the object  value
	:param i1:
	:param i2:
	:param p:
	:param s1:
	:param s2:
	:param s:
	:param sum:
	:param N0:
	:param E:
	:return:
	'''
	objecFunc = 0
	# symbol error probility
	# part1 decode s[i1] to the other points
	for i in range(12):
		if (i != i1):
			objecFunc += PS_err(s[i], s1, p, N0,i1,i2) * p[i1][0]

	# part1 decode s[i2] to the other
	for j in range(12):
		if (j != i2):
			objecFunc += PS_err(s[j],s2, p, N0, i1,i2) * p[i2][0]

	# part3 decode the others points to s[i1] and s[i2]
	for k in range(12):
		if (k != i2 and k != i1):
			objecFunc += (PS_err(s1, s[k], p, N0,i1,i2) * p[k][0] + PS_err(s2, s[k], p, N0,i1,i2) * p[k][0])

	return objecFunc


def optimizePairePoint(i1, i2, p, s, sum, N0, E):
	"""
	optimize s1,s2
	:param i1: the index of selected point1
	:param i2: the index of selected
	:param p: the probility of each symnol
	:param s: the coordinate of symbol
	:param sum
	:param N0
	:param E average power constarin
	:return:
	"""
	# cal b :the mean of rest point
	res = p*s
	b = np.array([0.0,0.0],dtype=float)
	d=0
	for i in range(12):
		if(i != i1 & i != i2):
			b += res[i]
			d += p[i][0]*(s[i][0]**2 + s[i][1]**2)

	#b -= sum

	a = -b / p[i1][0]
	c = p[i2][0] / p[i1][0] # then will use to cal s1 coordinate

	# search s2 in a cycle

	# 圆心坐标
	o1 = (p[i1][0] * a[0]) / (p[i1][0] + p[i2][0])
	o2 = (p[i1][0] * a[1]) / (p[i1][0] + p[i2][0])
	# 圆半径平方
	r2 = ((p[i1][0] * (E - d)) / (p[i2][0] * (p[i1][0] + p[i2][0]))) - \
		(p[i1][0]**3 / (p[i2][0] * np.power(p[i1][0] + p[i2][0],2))) * (a[0]**2 + a[1]**2)

	theta = [i*0.1*np.pi for i in range(0, 20)]

	#s2_tmp = [0,0] # initial position
	obj = 100
	s1 = s[i1] #np.array([0,0],dtype=float)
	s2 = s[i2] #np.array([0,0],dtype=float)
	s1_tmp = np.array([0,0],dtype=float)
	s2_tmp = np.array([0,0],dtype=float)


	for theta_i in theta:
		# the new pos of s2_tmp
		#s2x = o1+np.sqrt(r2)*np.cos(theta_i)
		#s2y = o2+np.sqrt(r2)*np.sin(theta_i)
		if r2<0:
			break
		s2_tmp = np.array([o1+np.sqrt(r2)*np.cos(theta_i), o2+np.sqrt(r2)*np.sin(theta_i)])
		#print("s2",s2_tmp)
		# the new pos of s1_tmp
		s1_tmp = a - c*s2_tmp
		#print("s1",s1_tmp,"s1",s1_tmp)
		# cal obj-func
		obj_i = objectFunc(i1,i2,p,s1_tmp,s2_tmp,s,N0)
		print("s1",s1_tmp,"s1",s1_tmp,"obj",obj_i)

		#obj.append(obj_i)
		if obj_i<obj:
			obj = obj_i
			s1 = s1_tmp
			s2 = s2_tmp

	print("min obj in ",i1,i2, obj,s1,s2)

	return s1,s2


def PaireOptimizeRecursize(p,s,sum,N0,E,iterationNum):
	"""
	the iteration process of algorithm
	:param p:
	:param s:
	:param sum:
	:param N0:
	:param E:
	:return:
	"""
	for i in range(iterationNum):
		index = [0,1,2,3,4,5,6,7,8,9,10,11]
		id = sample(index,2)
		# optimize the selected point
		s1, s2 =optimizePairePoint(id[0],id[1],p,s,sum,N0,E)
		# update the point with new pos
		s[id[0]] = s1
		s[id[1]] = s2

	return s


def main():
	#checkConstarin1(12)

	p,s,sum =getInitP_symbol(0.56)
	#p,s = getInitP_symbol8()
	E1 = 10
	E = 0

	for i in range(12):
		#if(i != i1 & i != i2):
		#b += res[i]
		E += p[i][0]*(s[i][0]**2 + s[i][1]**2)


	N0 = 2
	s_new = PaireOptimizeRecursize(p,s,sum, N0,E1, 2000)


	x = []
	y = []
	for i in range(12):
		x.append(s_new[i][0])
		y.append(s_new[i][1])

	plt.scatter(x,y,s=32,c='g')
	plt.grid()
	plt.show()


if __name__ == "__main__":
	main()
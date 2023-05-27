import numpy as np
# import string, random, os

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

class Poisson2:
    def __init__(self,):
        self.M = 2
        self.dim = 5
        
        self.fidelity_list = [16,32,128]

        self.bounds = ((0.1, 0.9),(0.1, 0.9),(0.1, 0.9),(0.1, 0.9),(0.1, 0.9))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
        
    def query(self, X, m):
        y_list = []
        N = X.shape[0]
        for n in range(N):
            y = self.single_query(X[n], m)
            y_list.append(y)
        #
        
        Y = np.array(y_list)
        re_Y = np.reshape(Y, [N,-1])
        
        #print(re_Y.shape)
        
        return re_Y 

    def single_query(self, X, m, interp=False):
        
        X = np.squeeze(X)
        fidelity = self.fidelity_list[m]
        u = self._poisson_solver(self.fidelity_list[m], X[0], X[1], X[2], X[3], X[4])
            
        return u

    def _poisson_solver(self, fidelity,u_0_x,u_1_x,u_y_0,u_y_1,u_dirac):
        x = np.linspace(0,1,fidelity)
        dx = x[1]-x[0]
        y = np.linspace(0,1,fidelity)
        u = np.zeros((fidelity+2,fidelity+2)) # Initial u used to create the b-vector in Ax = b
        # BC's and dirac delta
        u[0,:] = u_0_x
        u[-1,:] = u_1_x
        u[:,0] = u_y_0
        u[:,-1] = u_y_1
        if fidelity%2 == 0:
            u[int((fidelity+2)/2-1):int((fidelity+2)/2+1),int((fidelity+2)/2-1):int((fidelity+2)/2)+1] = u_dirac
        else:
            u[int((fidelity+1)/2),int((fidelity+1)/2)] = u_dirac

        # 5-point scheme
        A = np.zeros((fidelity**2,fidelity**2))
        for i in range(fidelity**2):
            A[i,i] = 4
            if i < fidelity**2-1:
                if i%fidelity != fidelity-1:
                    A[i,i+1] = -1
                if i%fidelity != 0 & i-1 >= 0:
                    A[i,i-1] = -1
            if i < fidelity**2-fidelity:
                A[i,i+fidelity] = -1
                if i-fidelity >= 0:
                    A[i,i-fidelity] = -1

        # Boundry conditions
        g = np.zeros((fidelity,fidelity))
        for i in range(1,fidelity+1):
            for j in range(1,fidelity+1):
                g[i-1,j-1] = u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1]

        b = dx**2*g.flatten()
        #x = np.linalg.solve(A,b)
        #u = x.reshape(fidelity,fidelity)

        # Sparse solver
        A_s = csc_matrix(A, dtype=float) # s for sparse
        b_s = csc_matrix(b, dtype=float)
        x_s = spsolve(A_s,b_s.T)
        u_s = x_s.reshape(fidelity,fidelity)

        return u_s

class Poisson3:
    def __init__(self,):
        self.M = 3
        self.dim = 5
        
        self.fidelity_list = [16,32,64,128]
#         self.fidelity_list = [16,32,64,72]

        self.bounds = ((0.1, 0.9),(0.1, 0.9),(0.1, 0.9),(0.1, 0.9),(0.1, 0.9))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]

    def query(self, X, m):
        y_list = []
        N = X.shape[0]
        for n in range(N):
            y = self.single_query(X[n], m)
            y_list.append(y)
        #
        
        Y = np.array(y_list)
        re_Y = np.reshape(Y, [N,-1])
        
        #print(re_Y.shape)
        
        return re_Y 
        
    def single_query(self, X, m, interp=False):
        
        X = np.squeeze(X)
        u = self._poisson_solver(self.fidelity_list[m], X[0], X[1], X[2], X[3], X[4])
            
        return u

    def _poisson_solver(self, fidelity,u_0_x,u_1_x,u_y_0,u_y_1,u_dirac):
        x = np.linspace(0,1,fidelity)
        dx = x[1]-x[0]
        y = np.linspace(0,1,fidelity)
        u = np.zeros((fidelity+2,fidelity+2)) # Initial u used to create the b-vector in Ax = b
        # BC's and dirac delta
        u[0,:] = u_0_x
        u[-1,:] = u_1_x
        u[:,0] = u_y_0
        u[:,-1] = u_y_1
        if fidelity%2 == 0:
            u[int((fidelity+2)/2-1):int((fidelity+2)/2+1),int((fidelity+2)/2-1):int((fidelity+2)/2)+1] = u_dirac
        else:
            u[int((fidelity+1)/2),int((fidelity+1)/2)] = u_dirac

        # 5-point scheme
        A = np.zeros((fidelity**2,fidelity**2))
        for i in range(fidelity**2):
            A[i,i] = 4
            if i < fidelity**2-1:
                if i%fidelity != fidelity-1:
                    A[i,i+1] = -1
                if i%fidelity != 0 & i-1 >= 0:
                    A[i,i-1] = -1
            if i < fidelity**2-fidelity:
                A[i,i+fidelity] = -1
                if i-fidelity >= 0:
                    A[i,i-fidelity] = -1

        # Boundry conditions
        g = np.zeros((fidelity,fidelity))
        for i in range(1,fidelity+1):
            for j in range(1,fidelity+1):
                g[i-1,j-1] = u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1]

        b = dx**2*g.flatten()
        #x = np.linalg.solve(A,b)
        #u = x.reshape(fidelity,fidelity)

        # Sparse solver
        A_s = csc_matrix(A, dtype=float) # s for sparse
        b_s = csc_matrix(b, dtype=float)
        x_s = spsolve(A_s,b_s.T)
        u_s = x_s.reshape(fidelity,fidelity)

        return u_s

class Heat2:
    def __init__(self,):
        self.M = 2
        self.dim = 3
        
        self.fidelity_list = [16,32,100]

        self.bounds = ((0,1),(-1,0),(0.01,0.1))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]

    def query(self, X, m):
        y_list = []
        N = X.shape[0]
        for n in range(N):
            y = self.single_query(X[n], m)
            y_list.append(y)
        #
        
        Y = np.array(y_list)
        re_Y = np.reshape(Y, [N,-1])
        
        #print(re_Y.shape)
        
        return re_Y
        
    def single_query(self, X, m, interp=False):

        X = np.squeeze(X)
        fidelity = self.fidelity_list[m]
        u, x = self._heat_solver(fidelity, X[2], X[0], X[1])
            
        return u
    
    def _thomas_alg(self, a, b, c, d):
        n = len(b)
        x = np.zeros(n)
        for k in range(1,n):
            q = a[k]/b[k-1]
            b[k] = b[k] - c[k-1]*q
            d[k] = d[k] - d[k-1]*q
        q = d[n-1]/b[n-1]
        x[n-1] = q
        for k in range(n-2,-1,-1):
            q = (d[k]-c[k]*q)/b[k]
            x[k] = q
        return x
    
    def _heat_solver(self, fidelity,alpha,neumann_0,neumann_1):
        x = np.linspace(0,1,fidelity)
        t = np.linspace(0,1,fidelity)
        u = np.zeros((fidelity+1,fidelity+2))
        dx = x[1]-x[0]
        dt = t[1]-t[0]

        # Set heaviside IC
        for i in range(fidelity):
            if i*dx >= 0.25 and i*dx <= 0.75:
                u[0,i+1] = 1

        for n in range(0,fidelity): # temporal loop
            a = np.zeros(fidelity); b = np.zeros(fidelity); c = np.zeros(fidelity); d = np.zeros(fidelity)
            for i in range(1,fidelity+1): # spatial loop
                # Create vectors for a, b, c, d
                a[i-1] = -alpha*dt/dx**2
                b[i-1] = 1+2*alpha*dt/dx**2
                c[i-1] = -alpha*dt/dx**2
                d[i-1] = u[n,i]

            # Neumann coniditions 
            d[0] = (d[0] - ((alpha*dt/dx**2)*2*dx*neumann_0))/2 # Divide by 2 to keep symmetry
            d[-1] = (d[-1] + ((alpha*dt/dx**2)*2*dx*neumann_1))/2
            a[0] = 0
            b[0] = b[0]/2
            c[-1] = 0
            b[-1] = b[-1]/2

            # Solve
            u[n+1,1:-1] = self._thomas_alg(a,b,c,d)
        v = u[1:,1:-1]
        return v, x

class Heat3:
    def __init__(self,):
        self.M = 3
        self.dim = 3
        
        self.fidelity_list = [16,32,64,100]

        self.bounds = ((0,1),(-1,0),(0.01,0.1))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
        
    def query(self, X, m):
        y_list = []
        N = X.shape[0]
        for n in range(N):
            y = self.single_query(X[n], m)
            y_list.append(y)
        #
        
        Y = np.array(y_list)
        re_Y = np.reshape(Y, [N,-1])
        
        #print(re_Y.shape)
        
        return re_Y 
        
    def single_query(self, X, m, interp=False):

        X = np.squeeze(X)
        fidelity = self.fidelity_list[m]
        u, x = self._heat_solver(fidelity, X[2], X[0], X[1])
            
        return u
    
    def _thomas_alg(self, a, b, c, d):
        n = len(b)
        x = np.zeros(n)
        for k in range(1,n):
            q = a[k]/b[k-1]
            b[k] = b[k] - c[k-1]*q
            d[k] = d[k] - d[k-1]*q
        q = d[n-1]/b[n-1]
        x[n-1] = q
        for k in range(n-2,-1,-1):
            q = (d[k]-c[k]*q)/b[k]
            x[k] = q
        return x
    
    def _heat_solver(self, fidelity,alpha,neumann_0,neumann_1):
        x = np.linspace(0,1,fidelity)
        t = np.linspace(0,1,fidelity)
        u = np.zeros((fidelity+1,fidelity+2))
        dx = x[1]-x[0]
        dt = t[1]-t[0]

        # Set heaviside IC
        for i in range(fidelity):
            if i*dx >= 0.25 and i*dx <= 0.75:
                u[0,i+1] = 1

        for n in range(0,fidelity): # temporal loop
            a = np.zeros(fidelity); b = np.zeros(fidelity); c = np.zeros(fidelity); d = np.zeros(fidelity)
            for i in range(1,fidelity+1): # spatial loop
                # Create vectors for a, b, c, d
                a[i-1] = -alpha*dt/dx**2
                b[i-1] = 1+2*alpha*dt/dx**2
                c[i-1] = -alpha*dt/dx**2
                d[i-1] = u[n,i]

            # Neumann coniditions 
            d[0] = (d[0] - ((alpha*dt/dx**2)*2*dx*neumann_0))/2 # Divide by 2 to keep symmetry
            d[-1] = (d[-1] + ((alpha*dt/dx**2)*2*dx*neumann_1))/2
            a[0] = 0
            b[0] = b[0]/2
            c[-1] = 0
            b[-1] = b[-1]/2

            # Solve
            u[n+1,1:-1] = self._thomas_alg(a,b,c,d)
        v = u[1:,1:-1]
        return v, x

class Dataset: 
    def __init__(self, Domain, trial):
        
        self.Domain = Domain
        self.Mfn = {
            'Heat3':Heat3,
            'Heat2':Heat2,
            'Poisson2':Poisson2,
            'Poisson3':Poisson3,
            # 'Burgers':Burgers,
            # 'Navier': Navier,
            # 'Lbracket': Lbracket,
        }[Domain]()
                
        # data_path = os.path.join('data/__processed__', Domain)
        # data_filename = Domain+'_trial'+str(trial)+'.h5'
        
        # print(os.path.join(data_path,data_filename))
        
        # raw = load_from_h5(os.path.join(data_path,data_filename))

        # self.Ntrain_list = raw['Ntrain']
        # self.Ntest_list = raw['Ntest']
        
        # self.MF_X_train = raw['X_train_list']
        # self.MF_y_train = raw['y_train_list']
        
        # self.MF_X_test = raw['X_test_list']
        # self.MF_y_test = raw['y_test_list']
        
        # self.MF_input_dims = raw['input_dims_list']
        # self.MF_output_dims = raw['output_dims_list'] 
        # self.MF_ground_dim = raw['output_ground_dim']

        # if raw['domain'] != self.Domain:
        #     print('ERROR: dataset and Mfn is not consistent...')
        #     exit(0)
        #

    def get_N_bounds(self):
        # get the normalized bounds
        # scales = self.get_scales(m)
        # X_mean = scales['X_mean']
        # X_std = scales['X_std']
        
        # N_lb = (self.Mfn.lb - X_mean)/X_std
        # N_ub = (self.Mfn.ub - X_mean)/X_std
        
        return self.Mfn.lb, self.Mfn.ub
    
    # def query(self, X, m):

    #     ym = self.Mfn.query(X, m)
        
    #     return ym
    
    def multi_query(self, N_X_query, m, x_mean, x_std):
        # enter the normalized Xquery
        y_query_list = []
        X_query = N_X_query*x_std + x_mean
        
        # maually clip the samples out of bounds
        # X_query = np.clip(X_query, self.Mfn.lb, self.Mfn.ub)
        # for i in range(len(X_query)):
        #     y_query = self.query(X_query[i], m)
        #     y_query_list.append(y_query)
        # y_query = np.stack(y_query_list,0)

        y_query = self.Mfn.query(X_query, m)
        # print(y_query.shape)

        return y_query


import numpy as np 
import time 
import warnings 
warnings.filterwarnings("default") 

class CGD():
    def __init__(self, objective, gradient, get_param_err, bounds, y, tol=1e-8, max_iter=1400, min_interval=1e-8, obj_diff_tol=1e-40, logger=None, normalize_direction=True, inc1=2, inc2=2):
        '''
        Create a conjugate gradient descent optimizer with line search.
        '''
        # set hyperparameters for gradient descent
        self.tol = tol # tolerance for gradient norm
        self.obj_diff_tol = obj_diff_tol # tolerance for change in objective functions
        self.max_iter = max_iter # maximal number of iterations

        # set hyperparameters for line search
        self.min_interval = min_interval # minimal interval used in line search
         
        # set functions and data 
        self.objective = objective 
        self.gradient = gradient
        self.get_param_err = get_param_err
        
        
        self.y = y # observation
        self.bounds = bounds # bounds for parameters 
        self.logger = logger  
        self.normalize_direction = normalize_direction
        self.inc1 = inc1 
        self.inc2 = inc2 

        return
    
    def handle_bds(self, pr, param, direction, eps):
        ratio = np.ones(len(pr))
        inbound = True 
        buffer = 1e-6
         
        for i in range(len(pr)):
            if pr[i] < self.bounds[0]:  
                if self.bounds[0] + buffer - pr[i] > 0:
                    ratio[i] = 0
                else:
                    ratio[i] = np.abs((self.bounds[0] + buffer - pr[i]) / (eps * direction[i]))
                inbound = False
            if pr[i] > self.bounds[1]: 
                if self.bounds[1] - buffer - pr[i] < 0:
                    ratio[i] = 0
                else:
                    ratio[i] = np.abs((self.bounds[1] - buffer - pr[i]) / (eps * direction[i]))
                inbound = False
        min_ratio = min(ratio)
        pr = param + min_ratio * eps * direction
        return pr, inbound, min_ratio


    def find_U_right(self, param, direction, err_l, err, err_r, left, middle, right, eps, inbound=True):
        '''
        Find a U shape along one direction
        err: err(param)
        err_r: err(right parameter)
        err_l: err(left parameter)
        '''  
        if not inbound: 
            return [err_l, err, err_r, left, middle, right, inbound]
        
        if err < err_l and err < err_r:  
            return [err_l, err, err_r, left, middle, right, inbound]

        else:
            pr = param + (right + eps) * direction 
            pr, inbound, min_ratio = self.handle_bds(pr, param, direction, right + eps)
            rr = min_ratio * (right + eps)
             
            if rr < right: 
                return [err_l, err, err_r, left, middle, right, inbound] 
            err_rr = self.objective(param + rr * direction, self.y)[0]
             
            if err < err_r and err_r < err_rr:  
                return [err_l, err, err_r, left, middle, right, inbound]
            if err_rr < 0.5:
                new_eps = eps*self.inc1 
            else:
                new_eps = eps*self.inc2 
            res_r = self.find_U_right(param, direction, 
                               err, err_r, err_rr, 
                               middle, right, rr, new_eps, inbound)  
            return res_r
        
    def find_U_left(self, param, direction, err_l, err, err_r, left, middle, right, eps, inbound=True):
        '''
        Find a U shape along one direction
        err: err(param)
        err_r: err(right parameter)
        err_l: err(left parameter)
        ''' 
        if not inbound: 
            return [err_l, err, err_r, left, middle, right, inbound]
    
        if err < err_l and err < err_r: 
            return [err_l, err, err_r, left, middle, right, inbound]
        else:
            pl = param + (left - eps) * direction 
            pl, inbound, min_ratio = self.handle_bds(pl, param, direction, left - eps)
            ll = min_ratio * (left - eps)
             
            if ll > left: 
                return [err_l, err, err_r, left, middle, right, inbound] 
            err_ll = self.objective(pl, self.y)[0]

            if err < err_l and err_l < err_ll:  
                return [err_l, err, err_r, left, middle, right, inbound]
            if err_ll < 0.5:
                new_eps = eps*self.inc1 
            else:
                new_eps = eps*self.inc2 
            res_l = self.find_U_left(param  , direction,
                                err_ll, err_l, err,
                                ll, left, middle, new_eps, inbound)
            return res_l 
          

    def line_search(self, param, direction, err):
        '''
        Find the minimizing parameter along the search direction.
        ''' 
         
        if np.linalg.norm(direction) < 1e-2:
            eps = 1e-2
        else:
            eps = 1e-5
        eps = 1e-4

        # check bounds         
        p_right = param + eps * direction 
        p_left = param - eps * direction 

        inbound = False 
        while eps > 1e-10:  
            if (not self.check_bd(p_right)) or (not self.check_bd(p_left)):
                eps /= 2
                p_right = param + eps * direction 
                p_left = param - eps * direction
                 
            else:
                inbound = True
                break
        
        if not inbound:
            print('parameter is out of bounds')
            print(direction)
            print('p right', p_right)
            print('p left', p_left) 
            return 0, inbound
         
        err_l = self.objective(param - eps * direction, self.y)[0] 
        err_r = self.objective(param + eps * direction, self.y)[0] 
        res_r = self.find_U_right(param, direction, err_l, err, err_r, -eps, 0, eps, eps, inbound=True) 
        res = res_r
        if (not (res_r[1] < res_r[0] and res_r[1] < res_r[2])) or res_r[4] == 0:
            res_l = self.find_U_left(param, direction, err_l, err, err_r, -eps, 0, eps, eps, inbound=True) 
            if res_l[1] < res_r[1]:
                res = res_l

        err_l, err, err_r, left, middle, right, inbound = res 
        alpha = self.bisect(err, err_l, err_r, left, right, middle, param, direction) 
        return alpha, inbound
    
    def bisect(self, err, err_l, err_r, left, right, middle, param, direction):
        max_iter = 15
        step = 0
        while True:
            if step == max_iter:
                break
            if right - left <= self.min_interval:
                break 
            # fit a quadratic
            A = np.array([
                [left ** 2, left, 1],
                [middle ** 2, middle, 1],
                [right ** 2, right, 1]
            ])
            y = np.array([err_l, err, err_r])
            coeff = np.linalg.inv(A) @ y
            a, b = coeff[:2]
            mid = - b / (2 * a)
            if mid < left or mid > right:
                mid = (left + right) / 2
            if mid == middle:
                break
            new_mid_err = self.objective(param+mid*direction, self.y)[0]
            if new_mid_err < err:
                if middle < mid:
                    left = middle 
                    middle = mid
                    err = new_mid_err  
                else:
                    right = middle
                    middle = mid 
                    err = new_mid_err
            elif new_mid_err > err:
                if middle < mid:
                    #print("new mid err > err | middle < mid => right = mid")
                    right = mid 
                    err_r = new_mid_err
                else:
                    #print("new mid err > err | middle > mid => left = mid")
                    left = mid 
                    err_l = new_mid_err
            else:
                break
            step += 1
        alpha = middle
        return alpha

    def ternery_search(self, left, right, param, direction):
        mid1 = 2 * left / 3 + right / 3
        mid2 = left / 3 + 2 * right / 3
        max_iter = 15
        step = 0
        while True:
            if step == max_iter:
                break
            if right - left <= self.min_interval:
                break 
            if mid2 - mid1 == right - left:
                break
            mid1 = 2 * left / 3 + right / 3
            mid2 = left / 3 + 2 * right / 3
            if mid1 <= left or mid2 >= right:
                break
            temp1 = param + mid1 * direction
            temp2 = param + mid2 * direction
            mid1_err = self.objective(temp1, self.y)[0]
            mid2_err = self.objective(temp2, self.y)[0]
            #self.logger.info(f"mid1_err {mid1_err} mid2_err {mid2_err}")
            if mid1_err < mid2_err:
                right = mid2   
            elif mid1_err > mid2_err:
                left = mid1
            else:
                left = mid1
                right = mid2
            step += 1
        alpha = 0.5 * (left + right)
        return alpha

    def check_bd(self, param):
        return np.all(self.bounds[0] <= param) and np.all(self.bounds[1] >= param) 
    
    def run(self, initial, gt, true_x):
        
 
        eps_state = 1e-5
        t1 = time.time()
        param = initial
        alpha = 0  
        err, xhat = self.objective(param, self.y)
        
        gradient = self.gradient(param, self.y, xhat)
        
        param_err = self.get_param_err(param, gt) 
        state_err = np.mean(np.square(true_x[np.abs(true_x) > eps_state] - xhat[np.abs(true_x) > eps_state]))
        percent_state_err = np.mean(np.abs((true_x[np.abs(true_x) > eps_state] - xhat[np.abs(true_x) > eps_state]) / true_x[np.abs(true_x) > eps_state])) * 100
        
        if not self.check_bd(param):
            print(param)
            print('Return because parameter is not in bound')
            return param, err, param_err, percent_state_err   
        direction = - gradient.copy() 
        k = 0          
        self.logger.info(f"gradient norm {np.linalg.norm(gradient)}")
        self.logger.info(f'gradient  {[gradient[i] for i in range(len(gradient))]}')
        self.logger.info(f'direction  {[direction[i] for i in range(len(direction))]}')
        msg = f'Step {0} | Loss {err:e} | Parameter Error {param_err:e} | State MSE {state_err:e} | Percent State Err {percent_state_err}%'
        self.logger.info(msg)
        msg = " ".join(str(p) for p in param)
        self.logger.info(f'param: {msg}') 
        j = 0
        
        all_param = []
        
        while True:
            all_param.append(param)
            
            if j == self.max_iter:
                break 
             
            if np.all(direction == 0):
                self.logger.info('Return because direction == 0.')
                break
            
            if k == len(direction):
                 
                direction = - gradient.copy()
                k = 0
                continue
             
            direction = - gradient.copy() 
            # whether to use normalized direction
            if self.normalize_direction:
                d_norm = direction / np.linalg.norm(direction)
            else:
                d_norm = direction.copy()
            
            alpha,inbound = self.line_search(param, d_norm, err)  
            if alpha == 0 and np.all(direction == -gradient):
                self.logger.info('\ndparam = 0 and direction=-gradient. Will set no change to parameter\n')
                break
             
            self.logger.info(f'    [gradient]    {[gradient[i] for i in range(len(gradient))]}')
            self.logger.info(f'    [direction]   {[direction[i] for i in range(len(direction))]}')
            self.logger.info(f'    [alpha]       {alpha}')
           
            
            dparam = alpha * d_norm
            if np.all(dparam) == 0:
                self.logger.info('\ndparam = 0. Will set no change to parameter\n')
 
            if np.linalg.norm(gradient) < self.tol:
                self.logger.info('Return because of small gradient')
                break
            
            
            param = param + dparam
            
             
            prev_grad = gradient.copy()
            err, xhat = self.objective(param, self.y)

            gradient = self.gradient(param, self.y, xhat)

            g_diff = gradient - prev_grad
            
            
            beta = np.dot(gradient, g_diff) / np.dot(prev_grad, prev_grad)
            direction = - gradient + beta * direction
                
            k += 1
            param_err = self.get_param_err(param, gt)
            state_err = np.mean(np.square(true_x[np.abs(true_x) > eps_state] - xhat[np.abs(true_x) > eps_state]))
            percent_state_err = np.mean(np.abs((true_x[np.abs(true_x) > eps_state] - xhat[np.abs(true_x) > eps_state]) / true_x[np.abs(true_x) > eps_state])) * 100
            msg = f'Step {j+1} | Loss {err:e} | Parameter Error {param_err:e} | State MSE {state_err:e} | Percent State Err {percent_state_err}%'
            self.logger.info(msg)
            msg = " ".join(str(p) for p in param)
            self.logger.info(f'param: {msg}')
            j+=1
           
        
        t2 = time.time()
        param_err = self.get_param_err(param, gt)
        state_err = np.mean(np.square(true_x[np.abs(true_x) > eps_state] - xhat[np.abs(true_x) > eps_state]))
        percent_state_err = np.mean(np.abs((true_x[np.abs(true_x) > eps_state] - xhat[np.abs(true_x) > eps_state]) / true_x[np.abs(true_x) > eps_state])) * 100
        
        msg = f'Step {j} | Loss {err:e} | Parameter Error {param_err:e} | State MSE {state_err:e} | Percent State Err {percent_state_err}%'
        self.logger.info(msg)
        self.logger.info(f'gradient  {[gradient[i] for i in range(len(gradient))]}')
        self.logger.info(f'direction  {[direction[i] for i in range(len(direction))]}')
        self.logger.info(f'alpha  {alpha}')
        runtime = f'Runtime {t2-t1:.4f}'
        self.logger.info(runtime)


        all_param = np.asarray(all_param)
        self.all_param = all_param

        return param, err, param_err, percent_state_err  
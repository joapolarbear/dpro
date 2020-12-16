import numpy as np
import tensorflow as tf
from logger_utils import Singleton, SingleLogger

def predict_error(_list, _list_pred):
    _list_pred = np.array(_list_pred)
    _list = np.array(_list)
    if len(_list) == 0:
        return None, "Original time is too small. Ignore!!!"

    diff = np.abs(_list_pred - _list) / _list
    return diff, "%f %%"%(np.average(diff * 100))

class DNNPredictor:
    def __init__(self, train_x, train_y, test_x, test_y, headers):
        self.headers = headers
        self.train_x = self.array2dict(train_x)
        self.train_y = train_y
        self.test_x = self.array2dict(test_x)
        self.test_y = test_y
        self.batch_size = 8

        feature_columns = []
        for header in self.headers:
            feature_columns.append(tf.feature_column.numeric_column(header))

        # Build a DNNRegressor, with 2x20-unit hidden layers, with the feature columns
        # defined above as input.
        self.model = tf.estimator.DNNRegressor(hidden_units=[20, 20], 
            feature_columns=feature_columns,
            model_dir='.models/dnnregressor')

    def array2dict(self, a):
        assert a.shape[1] == len(self.headers)
        _dict = {}
        for idx, key in enumerate(self.headers):
            _dict[key] = a[:,idx]
        return _dict

    def train(self):
        train_input_fn = self.make_dataset(self.batch_size, self.train_x, self.train_y, True, 1000)
        test_input_fn = self.make_dataset(len(self.test_y), self.test_x, self.test_y)
        # Train the model.
        # By default, the Estimators log output every 100 steps.
        self.model.train(input_fn=train_input_fn, steps=100000)

        # # Hook to stop training if loss does not decrease in over 100000 steps.
        # hook = tf.estimator.experimental.stop_if_no_decrease_hook(self.model, "loss", 1000)
        # ops = tf.get_default_graph().get_operations()
        # logging_hook = tf.estimator.LoggingTensorHook({
        #     "loss" : self.model['loss'], 
        #     "prediction" : prediction}, every_n_iter=100)
        # train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, hooks=[hook, logging_hook])
        # eval_spec = tf.estimator.EvalSpec(input_fn=test_input_fn)
        # tf.estimator.train_and_evaluate(self.model, train_spec, eval_spec)
        

    def predict(self):
        test_input_fn = self.make_dataset(len(self.test_y), self.test_x, self.test_y)

        # Evaluate how the model performs on data it has not yet seen.
        eval_result = self.model.predict(input_fn=test_input_fn, yield_single_examples=False)
        ypreds = np.array(list(eval_result)[0]['predictions']).reshape(-1, 1)
        print(ypreds.shape, self.test_y.shape)
        dirr, ratio = predict_error(self.test_y, ypreds)

        # Convert MSE to Root Mean Square Error (RMSE).
        print("\n" + 80 * "*")
        print(ratio)
        print()

    def test(self):
        test_input_fn = self.make_dataset(len(self.test_y), self.test_x, self.test_y)

        # Evaluate how the model performs on data it has not yet seen.
        eval_result = self.model.evaluate(input_fn=test_input_fn)

        # The evaluation returns a Python dictionary. The "average_loss" key holds the
        # Mean Squared Error (MSE).
        average_loss = eval_result["average_loss"]

        # Convert MSE to Root Mean Square Error (RMSE).
        print("\n" + 80 * "*")
        print("\nRMS error for the test set: {}".format(average_loss**0.5))
        print("\nRelative error for the test set: {} %".format(100 * average_loss**0.5 / np.average(self.test_y)))
        print()

    def make_dataset(self, batch_sz, x, y, shuffle=False, shuffle_buffer_size=1000):
        """Create a slice Dataset from a pandas DataFrame and labels"""

        def input_fn():
            dataset = tf.data.Dataset.from_tensor_slices((x, y))
            if shuffle:
                dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_sz).repeat()
            else:
                dataset = dataset.batch(batch_sz)
            # return dataset.make_one_shot_iterator().get_next()
            return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

        return input_fn

class BayesPredictor:
    def __init__(self, train_x, train_y, test_x, test_y, headers):
        self.headers = headers
        self.train_data = self.ret_dict(train_x, train_y)
        self.test_data = self.ret_dict(test_x, test_y)

    def ret_dict(self, xdata, ydata):
        _dict = {}
        _dict['y'] = ydata
        assert xdata.shape[1] == len(self.headers)
        for idx, key in enumerate(self.headers):
            _dict[key] = xdata[:,idx]
        return _dict

    def train(self):
        import pymc3 as pm
        # Context for the model
        with pm.Model() as normal_model:
            
            # The prior for the data likelihood is a Normal Distribution
            family = pm.glm.families.Normal()
            
            # Creating the model requires a formula and data (and optionally a family)
            pm.GLM.from_formula('y ~ S_mul + S_add + S_in + S_out + S_wei', data=self.train_data, family=family)
            
            # Perform Markov Chain Monte Carlo sampling letting PyMC3 choose the algorithm
            normal_trace = pm.sample(2000, cores=1)

from scipy.optimize import curve_fit
wei1, wei2 = 1, 1
ADD_ADDITIONAL = True
LINEARITY = 'linear' # \in ['linear', log', 'exp', sigmoid', 'piecewise', 'max_linear', 'max_exp'] or None

# def cost_func(xs, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3):
#     '''
#     gflops:
#         We only need a relative value of gflops, 
#         i.e., if we know fp32's is twice of fp16's, we can just fp32's = 2 and fp16's = 1,
#         the scale is hidden in the a2 
#         x[0]: relative gflops
#         x[1]: num of multiplication
#         x[2]: num of addition
#         x[3]: input size
#         x[4]: output size
#         x[5]: weight size

#         if len(x) > 6, there are some additional information, e.g., kernel size for Conv2D
#     '''
#     gflops = xs[0]
#     S_mul = xs[1]
#     S_add = xs[2]
#     wei_S_all = a3 * xs[3] + a4 * xs[4] + a5 * xs[5]
#     wei_S_all2 = a6 * xs[3] + a7 * xs[4] + a8 * xs[5]
#     if ADD_ADDITIONAL:
#         ### [H, W, C, R, S, P, Q, K, batch_size, use_bias]
#         addtional_term = a9 * xs[4] * xs[6+9]
#     else:
#         addtional_term = 0
#     return (a1 * S_mul + b1 + addtional_term) / (a2 * gflops + b2) + wei_S_all / gflops + b3 + gflops * wei_S_all2

# lower_bounds = tuple([0]*9 + [-np.inf]*3)

# def cost_func_conv2d(xs, a1, a2, a3, a4, a5, b1):
#     # wei1 = 1 / (1 + a9 * np.exp(a8 - S_mul))
#     # wei2 = 1 - wei1
#     H, W, C, R, S, P, Q, K, batch_size, use_bias = xs[6:]
#     kernel_size = R
#     batch_size = S_mul / (K*P*Q*C*R*S)

#     # flops_ = G * (1 / (1 + np.exp(-S_mul)) - a6)
#     # flops_ = G * np.log(a6 * S_mul + a7)

class CurveFiter:
    def __init__(self, headers, op_type="conv", E1_=None, E2_=None):
        self.headers = headers
        self.popt = self.perr = None
        self.op_type = op_type
        self.Es = [E1_, E2_]
        self.FIT_FUNC = None
        self.load_fit_func()

    def load_fit_func(self):
        E1 = 1 if self.Es[0] is None else self.Es[0]
        E2 = 2 if self.Es[1] is None else self.Es[1]
        E1 = E2 = 0
        
        def cost_func_cast(xs, a1, b1):
            _, _, _, S_in, _, _ = xs[0:6]
            return a1 * S_in + b1
        lower_bounds_cast = tuple([0]*1 + [-np.inf]*1)

        if LINEARITY == 'linear':
            def cost_func_conv2d(xs, a1, a2, a3, a4, a5, b1):
                G, S_mul, S_add, S_in, S_out, S_wei = xs[0:6]
                H, W, C, R, S, P, Q, K, batch_size, use_bias = xs[6:]

                wei_S_all = a3 * S_in + a4 * S_out + a5 * S_wei
                addtional_term = S_out * use_bias
                flops_ = G
                bandwidth = G
                kernel_size = R
                # return wei1 * ((a1 * S_mul + a2 * (addtional_term)) / (kernel_size**0.75 * flops_)) + wei2 * (wei_S_all) / bandwidth + b1
                return wei1 * ((a1 * S_mul + a2 * (addtional_term)) / (kernel_size**E1 * flops_)) + wei2 * (kernel_size**E2) * (wei_S_all) / bandwidth + b1

            def cost_func_dense(xs, a1, a2, a3, a4, a5, b1):
                G, S_mul, S_add, S_in, S_out, S_wei = xs[0:6]
                C_in, C_out, batch_size = xs[6:]

                wei_S_all = a3 * S_in + a4 * S_out + a5 * S_wei
                flops_ = G
                bandwidth = G
                # return wei1 * ((a1 * S_mul + a2 * (addtional_term)) / (kernel_size**0.75 * flops_)) + wei2 * (wei_S_all) / bandwidth + b1
                return wei1 * ((a1 * S_mul) / (flops_)) + wei2 * (wei_S_all) / bandwidth + b1

            lower_bounds_conv = lower_bounds_dense = tuple([0]*5 + [-np.inf]*1)

        elif LINEARITY == 'exp':
            def cost_func_conv2d(xs, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, b1):
                G, S_mul, S_add, S_in, S_out, S_wei = xs[0:6]
                H, W, C, R, S, P, Q, K, batch_size, use_bias = xs[6:]

                addtional_term = S_out * use_bias
                # flops_ = G
                # bandwidth = G
                flops_ = np.power(G, a20)
                bandwidth = np.power(G, a21)
                
                kernel_size = R
                S_mul = np.power(batch_size, a6) *  np.power(kernel_size, a7) * np.power(P, a8) * np.power(C, a9) * np.power(K, a10)
                S_wei = np.power(kernel_size, a11) * np.power(C, a12) * np.power(K, a13) 
                S_in = np.power(batch_size, a14) * np.power(H, a15) * np.power(C, a16) 
                S_out = np.power(batch_size, a17) * np.power(P, a18) * np.power(K, a19) 
                wei_S_all = a3 * S_in + a4 * S_out + a5 * S_wei
                return  wei1 * ((a1 * S_mul + a2 * (addtional_term)) / (flops_)) + wei2 * (wei_S_all) / bandwidth + b1
            
            def cost_func_dense(xs, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, b1):
                G, S_mul, S_add, S_in, S_out, S_wei = xs[0:6]
                C_in, C_out, batch_size= xs[6:]
          
                # flops_ = G
                # bandwidth = G
                flops_ = np.power(G, a15)
                bandwidth = np.power(G, a16)

                ai = S_mul / (S_in + S_out + S_wei)

                # wei1 = 1. / (1 + np.exp(-1.0 * a2 * (ai - a17)))
                # wei2 = 1 - wei1

                S_mul = np.power(batch_size, a6) * np.power(C_in, a7) * np.power(C_out, a8) 
                S_wei = np.power(C_in, a9) * np.power(C_out, a10)
                S_in = np.power(batch_size, a11) * np.power(C_in, a12)
                S_out = np.power(batch_size, a13) * np.power(C_out, a14)
                wei_S_all = a3 * S_in + a4 * S_out + a5 * S_wei
                return  wei1 * ((a1 * S_mul) / (flops_)) + wei2 * (wei_S_all) / bandwidth + b1
            
            lower_bounds_conv = tuple([0]*21 + [-np.inf]*1)
            lower_bounds_dense = tuple([0]*17 + [-np.inf]*1)

        elif LINEARITY == 'log':
            # def cost_func_conv2d(xs, a1, a2, a3, a4, a5, a6, a7, b1):
            #     G, S_mul, S_add, S_in, S_out, S_wei = xs[0:6]
            #     H, W, C, R, S, P, Q, K, batch_size, use_bias = xs[6:]
            #     wei_S_all = a3 * S_in + a4 * S_out + a5 * S_wei
            #     addtional_term = S_out * use_bias
            #     kernel_size = R
            #     return (a1) * np.log(batch_size) + \
            #             a2 * np.log(kernel_size) + \
            #             (a3) * np.log(P) + \
            #             a4 * np.log(C) + \
            #             (a5) * np.log(K) + (a6) * np.log(H) + use_bias * a7 + b1
            # lower_bounds_conv = lower_bounds_dense = tuple([0]*7 + [-np.inf]*1)

            def cost_func_conv2d(xs, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1):
                G, S_mul, S_add, S_in, S_out, S_wei = xs[0:6]
                H, W, C, R, S, P, Q, K, batch_size, use_bias = xs[6:]
                wei_S_all = a3 * S_in + a4 * S_out + a5 * S_wei
                flops_ = G
                bandwidth = G
                addtional_term = S_out * use_bias
                kernel_size = R

                S_mul = a1 * np.log(batch_size) + a6 * np.log(kernel_size) + a7 * np.log(P) + a8 * np.log(C) + a9 * np.log(K)

                return wei1 * ((S_mul + a2 * (addtional_term)) / (flops_)) + wei2 * (wei_S_all) / bandwidth + b1

            lower_bounds_conv = lower_bounds_dense = tuple([0]*9 + [-np.inf]*1)
        
        elif LINEARITY == 'piecewise':
            def cost_func_conv2d(xs, a1, a2, a3, a4, a5, a6, a7, a8, b1):
                G, S_mul, S_add, S_in, S_out, S_wei = xs[0:6]
                wei_S_all = a3 * S_in + a4 * S_out + a5 * S_wei

                num_of_wave = 1 / (1 + np.exp(-1.0 * a1 * (a8*(1+np.sin(a2 * S_mul)) - a6)) + b1)
          
                flops_ = G
                bandwidth = G
                return  num_of_wave * a7 * flops_
            lower_bounds_conv = tuple([0]*8 + [-np.inf]*1)

            # def cost_func_dense(xs, a1, a2, a3, a4, b1):
            #     G, S_mul, S_add, S_in, S_out, S_wei = xs[0:6]
            #     num_of_wave = 1 / (1 + np.exp(-1.0 * a1 * (np.sin(a2 * S_mul) - a3)) + b1)
          
            #     flops_ = G
            #     bandwidth = G
            #     return  num_of_wave * a4 * flops_
            # lower_bounds_dense = tuple([0]*4 + [-np.inf]*1)

            def cost_func_dense(xs, a1, a2, a3, a4, a5, a6, b1):
                G, S_mul, S_add, S_in, S_out, S_wei = xs[0:6]
                C_in, C_out, batch_size= xs[6:]

                wei_S_all = a3 * S_in + a4 * S_out + a5 * S_wei
                flops_ = G
                bandwidth = G
                # return wei1 * ((a1 * S_mul + a2 * (addtional_term)) / (kernel_size**0.75 * flops_)) + wei2 * (wei_S_all) / bandwidth + b1
                return wei1 * ((a1 * np.ceil(batch_size*C_out / a6)*C_in) / (flops_)) + wei2 * (wei_S_all) / bandwidth + b1
            lower_bounds_dense = tuple([0]*6 + [-np.inf]*1)

        elif LINEARITY == 'max_linear':
            def cost_func_conv2d(xs, a1, a2, a3, a4, a5, a6, b1):
                G, S_mul, S_add, S_in, S_out, S_wei = xs[0:6]
                H, W, C, R, S, P, Q, K, batch_size, use_bias = xs[6:]

                wei_S_all = a3 * S_in + a4 * S_out + a5 * S_wei
                addtional_term = S_out * use_bias
                flops_ = G
                bandwidth = G
                kernel_size = R

                alpha = a6

                term1 = wei1 * ((a1 * S_mul + a2 * (addtional_term)) / (flops_))
                term2 =  wei2 * (wei_S_all) / bandwidth
                # maxterm = (term1 * np.exp(alpha * term1) + term2 * np.exp(alpha * term2)) / (np.exp(alpha * term1) + np.exp(alpha * term2))
                maxterm = np.log(np.exp(alpha * term1) + np.exp(alpha * term2)) / alpha
                return  maxterm + b1
            lower_bounds_conv = tuple([0]*6 + [-np.inf]*1)

            def cost_func_dense(xs, a1, a2, a3, a4, a5, a6, b1):
                G, S_mul, S_add, S_in, S_out, S_wei = xs[0:6]
                C_in, C_out, batch_size= xs[6:]

                wei_S_all = a3 * S_in + a4 * S_out + a5 * S_wei
                flops_ = G
                bandwidth = G

                alpha = a6

                term1 = wei1 * ((a1 * S_mul) / (flops_))
                term2 =  wei2 * (wei_S_all) / bandwidth
                # maxterm = (term1 * np.exp(alpha * term1) + term2 * np.exp(alpha * term2)) / (np.exp(alpha * term1) + np.exp(alpha * term2))
                maxterm = np.log(np.exp(alpha * term1) + np.exp(alpha * term2)) / alpha

                # return wei1 * ((a1 * S_mul + a2 * (addtional_term)) / (kernel_size**0.75 * flops_)) + wei2 * (wei_S_all) / bandwidth + b1
                return  maxterm + b1

            lower_bounds_dense = tuple([0]*6 + [-np.inf]*1)

        elif LINEARITY == 'max_exp':
            def cost_func_conv2d(xs, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, b1):
                G, S_mul, S_add, S_in, S_out, S_wei = xs[0:6]
                H, W, C, R, S, P, Q, K, batch_size, use_bias = xs[6:]

                addtional_term = S_out * use_bias
                # flops_ = G
                # bandwidth = G
                flops_ = np.power(G, a20)
                bandwidth = np.power(G, a21)
                
                kernel_size = R
                S_mul = np.power(batch_size, a6) *  np.power(kernel_size, a7) * np.power(P, a8) * np.power(C, a9) * np.power(K, a10)
                S_wei = np.power(kernel_size, a11) * np.power(C, a12) * np.power(K, a13) 
                S_in = np.power(batch_size, a14) * np.power(H, a15) * np.power(C, a16) 
                S_out = np.power(batch_size, a17) * np.power(P, a18) * np.power(K, a19) 
                wei_S_all = a3 * S_in + a4 * S_out + a5 * S_wei

                alpha = a22
                term1 = wei1 * ((a1 * S_mul + a2 * (addtional_term)) / (flops_))
                term2 = wei2 * (wei_S_all) / bandwidth
                # maxterm = (term1 * np.exp(alpha * term1) + term2 * np.exp(alpha * term2)) / (np.exp(alpha * term1) + np.exp(alpha * term2))
                maxterm = np.log(np.exp(alpha * term1) + np.exp(alpha * term2)) / alpha
                return  maxterm + b1
            lower_bounds_conv = tuple([0]*22 + [-np.inf]*1)

            def cost_func_dense(xs, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, b1):
                G, S_mul, S_add, S_in, S_out, S_wei = xs[0:6]
                C_in, C_out, batch_size = xs[6:]
          
                # flops_ = G
                # bandwidth = G
                flops_ = np.power(G, a15)
                bandwidth = np.power(G, a16)

                ai = S_mul / (S_in + S_out + S_wei)

                # wei1 = 1. / (1 + np.exp(-1.0 * a2 * (ai - a17)))
                # wei2 = 1 - wei1

                S_mul = np.power(batch_size, a6) * np.power(C_in, a7) * np.power(C_out, a8) 
                S_wei = np.power(C_in, a9) * np.power(C_out, a10)
                S_in = np.power(batch_size, a11) * np.power(C_in, a12)
                S_out = np.power(batch_size, a13) * np.power(C_out, a14)
                wei_S_all = a3 * S_in + a4 * S_out + a5 * S_wei

                alpha = a18
                term1 = wei1 * ((a1 * S_mul) / (flops_))
                term2 = wei2 * (wei_S_all) / bandwidth
                # maxterm = (term1 * np.exp(alpha * term1) + term2 * np.exp(alpha * term2)) / (np.exp(alpha * term1) + np.exp(alpha * term2))
                maxterm = np.log(np.exp(alpha * term1) + np.exp(alpha * term2)) / alpha
                return  maxterm + b1
            lower_bounds_dense = tuple([0]*18 + [-np.inf]*1)

        else:
            raise

        if self.op_type == "conv" or self.op_type == "Conv2D":
            self.fit_func = cost_func_conv2d
            self.lower_bounds = lower_bounds_conv
        elif self.op_type == "dense" or self.op_type == "MatMul":
            self.fit_func = cost_func_dense
            self.lower_bounds = lower_bounds_dense
        elif self.op_type == "CastToFp16" or self.op_type == "CastToFp32":
            self.fit_func = cost_func_cast
            self.lower_bounds = lower_bounds_cast
        else:
            raise ValueError(self.op_type)
        self.up_bounds = tuple(len(self.lower_bounds) * [np.inf])
        self.p0 = [1]*len(self.lower_bounds)

        # if LINEARITY == 'max':
        #     self.p0[21] = 0.0001

    def no_linear_label(self, data_):
        if LINEARITY == 'linear':
            return data_
        if LINEARITY == 'exp':
            return data_
        elif LINEARITY == 'log':
            # return data_
            return np.log(data_ + 1)
        elif LINEARITY in ['piecewise', 'max_linear', 'max_exp'] :
            return data_
        else:
            raise ValueError("LINEARITY should be log:{}".format(LINEARITY))

    def train(self, train_x, train_y):
        if len(train_x) == 0:
            SingleLogger().warin("The size of training dataset is 0, skip...")
            self.popt = self.perr = None
        else:
            _train_x = np.transpose(train_x)
            _train_y = self.no_linear_label(np.transpose(train_y).flatten())
            # self.fit_func(_train_x, *self.p0)
            self.popt, pcov = curve_fit(self.fit_func, _train_x, _train_y, 
                bounds=(self.lower_bounds, self.up_bounds), p0=self.p0, maxfev=100000)
            self.perr = np.sqrt(np.diag(pcov))
        return self.popt, self.perr

    def test(self, test_x, test_y, verbose=True):
        if len(test_x) == 0:
            SingleLogger().warn("The size of test dataset is 0, skip...")
            return None
        _test_x = np.transpose(test_x)
        _test_y = self.no_linear_label(np.transpose(test_y).flatten())

        if self.popt is None:
            SingleLogger().error("Curvefitter is not trained")
        avgs_pred = self.fit_func(_test_x, *self.popt)
        diff, ratio = predict_error(_test_y, avgs_pred)
        error = float(ratio.split("%")[0])
        if verbose:
            SingleLogger().info("average error: %f %%"%(error))
        return error

    def predict(self, xdata):
        return self.fit_func(xdata, *self.popt)





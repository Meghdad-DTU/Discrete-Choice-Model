from scipy.optimize import minimize
from scipy.stats import norm
from numdifftools import Hessian, Gradient
from datetime import datetime
import halton as hl # Halton sequence
import lhsmdu # Latin hypercube sampling 
import time

class DCM_logit(object):
    
    def __init__(self, mixing=0, startingiterations=100, verbose=False, RUM=True):        
        self.mixing = mixing            # set to 1 for models that include random parameters    
        self.startingiterations = startingiterations
        self.verbose = verbose
        self.RUM = RUM                  # Random Utility Maximization is true otherwise Random Regret Minimization
        self.df = None
        self.finalLL = None             # final Log likelihood 
        self.no_parameters = None       # number of estimated parameters 
        self.estimates = None           # estimated parameters
        self.cov = None                 # variance-covariance matrix
                
    def Initialization(self, modelname:str, data, no_est:int, parnames=None, startvalues=None):
        self.modelname = modelname  
        self.data = data
        self.no_est = no_est     # Number of parametrs for estimation
        self.parnames = parnames # Parameter's name 
        self.startvalues = startvalues
        # Set starting values
        if startvalues is None:
            self.startvalues = np.repeat(0,self.no_est)
        
        nrow, ncol = data.shape
        # Determine number of individuals in the data
        self.N = len(set(data.ID))
        
        # Determine number of choice tasks in the data
        self.choicetasks = nrow
        
        # Set number of alternatives in model
        self.number_of_alts = len(set(self.data.choice))
        
        # Calculate LL(0) - this basic calculation assumes that availabilities are all 1        
        self.LL0 = nrow*np.log(1/number_of_alts)        
    
    # generate draws (using Halton, Latin or Uniform )
    def _halton(self,dim,sample):
        return hl.halton(dim,sample)
    
    def _latin(self,dim,sample):
        return np.asarray(lhsmdu.sample(sample,dim))
    
    def _uniform(self,dim,sample):
        return np.random.rand(sample*dim).reshape(sample,dim)  
    
    
    def _draw_transform(self,draws,randPars,transformPars,transform):
        """function changes the generated draws based on starting values"""
        
        # transformation of draws into standard normal distribution N(mean=0,std=1)
        draws_0_1= norm.ppf(draws)
        for rp in randPars:
            ind_rp = randPars.index(rp)
            items = [s for s in parNames if rp in s]
            ind_startVal = np.where(np.isin(parNames,items))[0]
            if len(ind_startVal)<2:
                continue
            ###################################
            # par1,par2:
            # for normal family par1:mean, par2:std
            # for sym_triangular, par1:a, par2:b
            ####################################
            par1,par2 = self.startvalues[ind_startVal]
            print(items, par1,par2)
        
            if transform=="normal":
                draws[:,ind_rp] = par1 + draws_0_1[:,ind_rp]*par2                               
            # positive lognormal
            elif transform=="pos_lognormal":
                draws[:,ind_rp] = np.exp(par1 + draws_0_1[:,ind_rp]*par2)               
            # negative lognormal
            elif transform=="neg_lognormal":
                draws[:,ind_rp] = -1*np.exp(par1 + draws_0_1[:,ind_rp]*par2) 
            # symmetrical triangular
            elif transform=="sym_triangular":
                U1 = self._uniform(1,self.Ndraws*self.N).flatten()
                U2 = self._uniform(1,self.Ndraws*self.N).flatten()
                draws[:,ind_rp] = par1 + (U1+U2)*par2
            else:
                print("transform is unknown, use normal, pos_lognormal, neg_lognormal or sym_triangular")
            
        return draws  
    
    
    def Random_Parameters(self, Ndraws, randPars:list, method="halton", transformPars=None, transform=None):
        """
        # Ndraws: 
        set number of draws to use per person and per parameter
        
        # randPars: 
        string list including the names of random parameters
        
        # method:
        draw methods including "halton", "latin" (Latin hypercube) or "uniform" 
        
        # transformPars:
        list of random parameters for draw transformation
         
        # transform:
        draws transformation: "normal","pos_lognormal","neg_lognormal","sym_triangular"        
        
        """
          
        # dimentions: define number of random terms in the model
        self.Ndraws=Ndraws
        dimentions=len(randPars)
        
        drawNameList=["halton","latin","uniform"]
        drawMethod=[self._halton,self._latin,self._uniform]
        
        if method not in drawNameList:
            raise Exception("There are only three methods to generate draws: halton, latin & uniform.")
        
        if transformPars is None:
            transformPars=randPars
            
        ind=drawNameList.index(method) 
        draws=drawMethod[ind](dimentions,Ndraws*self.N)
            
        # assign names to individual sets of draws - need one entry per dimension
        drawNames=["draws"+"_"+par for par in randPars] 
        
        transformed_draws = self._draw_transform(draws=draws,
                                               randPars=randPars,
                                               transformPars=transformPars,
                                               transform=transform)         
            
        # working copies
        draws_internal = pd.DataFrame(transformed_draws,columns=drawNames)
        data_internal = self.data.copy()
        
        # combine draws with estimation data
        # turn into datatable and add indices for people and draws, for merging later on
        draws_internal["ID"] = np.repeat(self.data.ID.unique(),Ndraws)
        draws_internal["draws_index"] = np.tile(range(1,Ndraws+1),self.N) 
        draws_internal["merge_index"] = draws_internal["ID"]+ draws_internal["draws_index"]/100000
        
        # reformat data_internal to accommodate draws_internal, expands datatable to 
        # replicate each row R times, where R=number of draws
        data_internal = pd.DataFrame(pd.np.repeat(data_internal.values,Ndraws,axis=0),columns=data_internal.columns)
        nrow,_ = data_internal.shape
        
        # add a column with index for draws_internal
        data_internal["draws_index"] = np.tile(range(1,Ndraws+1),int(nrow/Ndraws))        
        
        # add an index for merging  
        data_internal["merge_index"] = data_internal["ID"]+data_internal["draws_index"]/100000
        
        # merging data_internal and draws_internal
        merged_data_internal = pd.merge(data_internal,draws_internal,on=["merge_index","ID","draws_index"],how="inner")       
        
        # global versions
        self.draws = draws_internal
        self.data = merged_data_internal
    
    def Define_Model_Parameters(self):    
        # initial betas
        beta = self.startvalues    
        
        # These will need fine tuning depending on the model, etc
        lowerlimits=self.startvalues-0.1
        upperlimits=self.startvalues+0.1
        
        return beta,lowerlimits,upperlimits 
    
    ## Custom log-likelihood function
    def loglike(self, parameters, functionality=1, data=None):        
        data = self.data
        if data is not None:
            data = data
        
        # R: Regret, U: Utility
        RU = ["RU%d"%x for x in range(1,self.number_of_alts+1)]
        ERU = ["eRU%d"%x for x in range(1,self.number_of_alts+1)]
        
        #####################################################################
        # subsection required only in estimation                            #
        #####################################################################
       
        if functionality == 1: 
                    
            nrow, ncol = data.shape
            BETA =  parameters 
        
            if self.mixing==0:
                df = data.loc[:,["ID","choice","running_task"]]
            else:
                df = data.loc[:,["ID","choice","running_task","draws_index"]]
                
        
            ##########################################
            # define utility functions
            ##########################################
            for i,ru in enumerate(RU):
                df[ru] = RegUti_function(df, data, BETA)[i] 
                               
            # create a term we subtract from all utilities for numerical reasons:
            # (middle value between max and min of utilities)
            df["RUmax"] = df[RU].values.max(1)
            df["RUmin"] = df[RU].values.min(1)
            df["RUcensor"] = df[["RUmax","RUmin"]].values.mean(1)     
            
            for ru in RU:
                df[ru] = df[ru] - df["RUcensor"]           
        
            # exponentiate utilities
            ## for Random Utility Maximization
            if self.RUM: 
                for eru,ru in zip(ERU,RU):
                    df[eru] = np.exp(df[ru]) 
            ## for Random Regret Minimization
            else:
                for eru,ru in zip(ERU,RU):
                    df[eru] = np.exp(-1*df[ru])
         
            # calculate probability for chosen alternative for each observation 
            # (remember to take availabilities into account here if needed) 
            ## e.g. if choice = 1, p = eu1/(eu1+eu2)
            
            prob = np.repeat(0,df.shape[0])
            df["sum"] = np.sum(df[ERU],axis=1)
            
            for alt,col in zip(range(1, number_of_alts+1),ERU):
                prob = prob + np.where(df["choice"] == alt, df[col]/(df["sum"]), 0)
            df["P"] = prob            
                                   
            # compute log-likelihood, different approach with and without mixing
            if self.mixing==0:
            # take product across choices for the same person (likelihood)
            # then take the log for log-likelihood
                L = df[["ID","P"]].groupby("ID").prod()
                LL = np.log(L).values.flatten()
            
            else:
                # take product across choices for the same person and the same draw
                L = df[["ID","draws_index","P"]].groupby(["ID","draws_index"]).prod()
                # then average across draws and take the log for simulated log-likelihood                
                LL = np.log(L.groupby("ID").mean()).values.flatten()
            
            self.df = df    
            if self.verbose:
                print("Function value:", np.round(-1*sum(LL),3))
            
            return np.round(-1*sum(LL),3) # important: a negative log-likelihood since there is no maximize function in scipy  
    
        #####################################################################
        # subsection required only if producing predictions                 #
        #####################################################################
   
        if functionality==2:
            # for predictions, we need probabilities for all alternatives, not just the chosen one, again, remember availabilities if needed
            P = ["P%d"%x for x in range(1,self.number_of_alts+1)]
            for p,eru in zip(P,ERU):
                self.df[p] = self.df[eru]/self.df["sum"] 
     
            # copy part of the overall data table into a new datatable - ID, task, chosen alternative, probs and prob of chosen
            selected_cols = ["ID","running_task","choice","P"] + P
            probs_out = self.df.loc[:,selected_cols]
            # take mean across draws (only applies with mixing when multiple copies of same ID-running_task rows exist)
            probs_out = probs_out.groupby(["ID","running_task"]).mean()         
            return probs_out
    
        #####################################################################
        # subsection required only for outlier                            #
        #####################################################################
    
        if functionality==3:
        # compute log-likelihood, different approach with and without mixing
            if self.mixing==0:
                # take product across choices for the same person (likelihood)
                # then take the log for log-likelihood
                L = self.df[["ID","P"]].groupby("ID").prod()
                LL = np.log(L).values.flatten()
            else:
                # take product across choices for the same person and the same draw
                L = self.df[["ID","draws_index","P"]].groupby(["ID","draws_index"]).prod()
                # then average across draws and take the log for simulated log-likelihood                
                LL = np.log(L.groupby("ID").mean()).values.flatten()
            return LL 
        
    def initial_beta(self):
        beta,lowerlimits,upperlimits = self.Define_Model_Parameters()         
        
        if self.verbose:
            print("Searching for starting values")
        current_LL=-1*(self.loglike(beta))
        i=0   
        # Now iterate to try different values
        while(i<(self.startingiterations+1)):            
            betaVal= lowerlimits + np.random.random(len(lowerlimits))*(upperlimits-lowerlimits)
            beta_test = betaVal
            LL_test=-1*self.loglike(beta_test)
            if(LL_test>current_LL):
                current_LL = LL_test
                beta = beta_test
            i=i+1
            if self.verbose:
                print("Iteration %d: LL=%s (best LL so far: %s)"%(i,LL_test,current_LL)) 
        
        if self.verbose: 
            print("\nInitial model estimations:")
            for s,n in zip(self.parnames,np.round(beta,3)): 
                print(s,":",n)    
        return beta
    
    def correlation_from_covariance(self,covariance):
        v = np.sqrt(np.diag(covariance))
        outer_v = np.outer(v, v)
        correlation = covariance / outer_v
        correlation[covariance == 0] = 0
        return correlation  
    
    def fit(self,method='Nelder-Mead'):
        """
        other methods for optimization:
        'Powell','CG','BFGS','Newton-CG', etc.
        look at: scipy.optimize.minimize        
        """
        
        self.start_time = time.time()        
        self.initials = self.initial_beta()
        lik_model = minimize(self.loglike, self.initials, method=method)
                          
        #self.message = lik_model["success"]
        self.message = lik_model["message"]        
        self.finalLL = -1*lik_model["fun"]
        
        # Estimated betas
        self.est = lik_model['x']
        np.warnings.filterwarnings('ignore')
        Hfun = Hessian(self.loglike)
        hessian_ndt = Hfun(self.est)
        self.hess_inv = np.linalg.inv(hessian_ndt)
        
        #########################  important #########################################################
        ## If the diagonal of the inverse Hessian is negative, it is not possible to use square root.
        # In this case, you can calculate a pseudo-variance matrix. The pseudo-variance matrix is 
        # LL' with L=cholesky(H-1) with H being the Hessian matrix.
        ##############################################################################################
        
        diag_elements = np.diag(self.hess_inv)
        if (diag_elements <0).any():
            choleski_factorization = np.linalg.cholesky(self.hess_inv)
            se = np.sqrt(np.diag(choleski_factorization))
        else:
            se = np.sqrt(np.diag(self.hess_inv))
        
        self.estimates = pd.DataFrame({'est':self.est,
                                  'std err':se,
                                  "trat_0":self.est/se,
                                  "trat_1":(self.est -1)/se})
        
        self.estimates.index = self.parnames 
        self.cov = self.hess_inv
        self.covMat = pd.DataFrame(self.cov,columns=self.parnames).set_index(self.parnames)
        corr = self.correlation_from_covariance(self.cov)
        self.corrMat = pd.DataFrame(corr,columns=self.parnames).set_index(self.parnames)                         
        
        return "Model estimation is completed"  
    
        ## for outlier detection
    def outlier(self):
        unique_ID = np.unique(self.data["ID"].values) 
        task_per_ID = self.data["ID"].value_counts().values
        if self.mixing==0:
            self.Ndraws=1        
        P_per_task = np.exp(self.loglike(self.est,3))**(1/task_per_ID*self.Ndraws)
        return unique_ID, P_per_task
        
    def predict(self,data):    
        return self.loglike(parameters=self.est, functionality=2, data=data)     
   
    def output_print(self):        
        now = datetime.now()
        dt_string = now.strftime("%b-%d-%Y %H:%M:%S")
        print("Model run at:", dt_string ,"\n")   
        print("Model name: ",self.modelname,"\n")
        print("Model diagnosis:",self.message,"\n")
        print("Number of decision makers:",self.N)
        print("Number of observations:",self.choicetasks,"\n")
        if(self.LL0<0): 
            print("LL(0): ",np.round(self.LL0,2))
        print("LL(final): ",np.round(self.finalLL,2))
        self.no_parameters = len(self.parnames)
        print("Estimated parameters: ",self.no_parameters,"\n")
        if(self.LL0<0): 
            print("Rho-sq: ",np.round(1-(self.finalLL/self.LL0),2))
        if(self.LL0<0): print("Adj. rho-sq: ",np.round(1-((self.finalLL-len(self.parnames))/self.LL0),2))
        print("AIC: ",np.round(-2*self.finalLL+2*(len(self.parnames)),2))
        print("BIC: ",np.round(-2*self.finalLL+(len(self.parnames))*np.log(self.choicetasks),2),"\n")
        print("Time taken: ",time.strftime("%H:%M:%S", time.gmtime(np.round(time.time() - self.start_time,2))), "\n")
        print("Estimates:")
        print(self.estimates,"\n")
        print("Covariance matrix:")
        print(self.covMat,"\n")
        print("Correlation matrix:")
        print(self.corrMat,"\n")
        print("20 worst outliers in terms of lowest average per choice prediction:")
        print(pd.DataFrame(self.outlier()[1],self.outlier()[0], columns=["Av prob per choice"]).sort_values(by="Av prob per choice").iloc[:20,:],"\n")
        print("Changes in parameter estimates from starting values:")
        print(pd.DataFrame({"initials":self.initials, "est": self.est, "diff":self.est - self.initials}).set_index(self.parnames))
classdef halfCheetahEnv < rl.env.MATLABEnvironment
    % This code is based on Paulo Carvalho's code (2019)    
    properties
        % Select environment name here 
        open_env = py.gym.make('HalfCheetah-v2'); 
    end
    methods              
        function this =halfCheetahEnv()
            % Initialize Observation settings
            ObservationInfo             = rlNumericSpec([1 1]);
            ObservationInfo.Name        = 'HalfCheetah';
            ObservationInfo.Description = 'real number';         
            ActionInfo                  = rlFiniteSetSpec([0 1]); 
            ActionInfo.Name             = 'left, right';            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
        end        
        function [observation, reward, done, info] = step(this,Action)
            result      = this.open_env.step(Action);             
            observation = single(result{1}); 
            reward      = double(result{2});
            done        = result{3};
            info        = [];                 
        end
        function InitialObservation = reset(this)
            result             = this.open_env.reset();
            InitialObservation = single(result);
        end
        function screen = render(this)
            screen = this.open_env.render();
        end
        function close(this)
            this.open_env.close()
        end
    end
end

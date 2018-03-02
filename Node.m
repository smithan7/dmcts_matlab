classdef Node
    %UNTITLED3 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        x
        y
        i
        end_time = 100;
    end
    
    methods
        function obj = Node(xi, yi, ii)
            obj.x = xi;
            obj.y = yi;
            obj.i = ii;
        end
                
        function reward = get_reward(obj, t, impact_flag, coord)
            if ~exist('coord','var') || isempty(coord)
              impact_flag=0;
            end
            if impact_flag == 0
                reward = max(0, obj.end_time - t);
                return
            else
               [times, probs] = coord.get_claims(obj.i);
                reward = obj.get_reward(t);
                r_c = 0.0;
                for ii=1:length(times)
                    if times(ii) > t
                        r_c = r_c + (1.0 - probs(ii)) * max(obj.get_reward(times(ii), 0, coord), 0.0);
                    end
                end
                reward =reward - r_c; 
            end
        end
    end
end


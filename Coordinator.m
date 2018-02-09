classdef Coordinator
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        times
        probs
        n_nodes
    end
    
    methods
        % initialize
        function obj = Coordinator(n)
            obj.n_nodes = n;
            for i=1:n
                obj.times{i,1} = [0, inf];
                obj.probs{i,1} = [0, 1];
            end
        end
        
        % insert claim
        function obj = add_claim(obj, i, t, p)
            for ii=2:length(obj.times{i})
                if obj.times{i}(ii) > t && obj.times{i}(ii-1) < t
                    obj.times{i} = [obj.times{i}(1:ii-1), t, obj.times{i}(ii:end)];
                    obj.probs{i} = [obj.probs{i}(1:ii-1), min(1,p+obj.probs{i}(ii-1)), min(1,obj.probs{i}(ii:end)+p)];
                    return
                else
                    if obj.times{i}(ii-1) == t
                        obj.probs{i}(ii-1) = obj.probs{i}(ii-1) + p;
                    end
                end
            end
        end
        
        % get probability a task is taken at time
        function p_taken = get_p_taken(obj, i, t)
            p_taken = 0;
            for j=2:length(obj.times{i})
                if t < obj.times{i}(j) && t >= obj.times{i}(j-1)
                    p_taken =  obj.probs{i}(j-1);
                    return
                end
            end
        end
        
        function get_impact(obj, i, t)
            
        end
        
        % reset
        function obj = reset(obj)
            for i=1:obj.n_nodes
                obj.times{i,1} = [0, inf];
                obj.probs{i,1} = [0, 1];
            end
        end
    end
end


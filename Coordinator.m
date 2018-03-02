classdef Coordinator
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        times
        probs
        agent
        n_nodes
    end
    
    methods
        % initialize
        function obj = Coordinator(n)
            obj.n_nodes = n;
            for i=1:n
                obj.agent{i,1} = [-1, -1];
                obj.times{i,1} = [0, inf];
                obj.probs{i,1} = [0, 1];
            end
        end
        
        % insert claim
        function obj = add_claim(obj, i, t, p, ai)
            for ii=2:length(obj.times{i})
                if obj.times{i}(ii) > t && obj.times{i}(ii-1) < t
                    obj.times{i} = [obj.times{i}(1:ii-1), t, obj.times{i}(ii:end)];
                    obj.probs{i} = [obj.probs{i}(1:ii-1), min(1,p+obj.probs{i}(ii-1)), min(1,obj.probs{i}(ii:end)+p)];
                    obj.agent{i} = [obj.agent{i}(1:ii-1), ai, obj.agent{i}(ii:end)];
                    return
                else
                    if obj.times{i}(ii-1) == t && obj.agent{i}(ii-1) ~= ai
                        obj.probs{i}(ii-1) = obj.probs{i}(ii-1) + p;
                        if obj.agent{i}(ii-1) < 0
                            obj.agent{i}(ii-1) = ai;
                        end
                    end
                end
            end
        end
        
        % get probability a task is taken at time
        function p_taken = get_p_taken(obj, i, t, ai)
            p_taken = 0;
            if length(obj.times{i}) <= 2 || t == 0.0
                return
            end
            for j=2:length(obj.times{i})
                if t < obj.times{i}(j) && t >= obj.times{i}(j-1) %&& ai ~= obj.agent{i}(j-1)
                    p_taken =  obj.probs{i}(j-1);
                    return
                end
            end
        end
        
        function [t, p] = get_claims(obj, ii)
            t = obj.times{ii,:};
            p = obj.probs{ii,:};    
        end
        
        % reset
        function obj = reset(obj, an)
            for i=1:obj.n_nodes
                obj.times{an,i,1} = [0, inf];
                obj.probs{an,i,1} = [0, 1];
                obj.agent{an,i,1} = [-1,-1];
            end
        end
    end
end


classdef Node
    %UNTITLED3 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        x
        y
        end_time = 100;
    end
    
    methods
        function obj = Node(xi, yi)
            obj.x = xi;
            obj.y = yi;
        end
                
        function reward = get_reward(obj, t)
            reward = max(0, obj.end_time - t);
        end
        
    end
    
end


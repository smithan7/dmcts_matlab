classdef World
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        n_nodes
        nodes
        travel
        node_status
    end
    
    methods
        %% initilalize
        function G = World(n_nodes)
            G.node_status = ones(1,n_nodes);
            G.n_nodes = n_nodes;
            for i=1:n_nodes
                n = Node(10*randn() - 5, 10*randn() -5);
                G.nodes{i} = n;
            end
            G.travel = zeros(n_nodes, n_nodes);
            for i=1:n_nodes
                for j=1:n_nodes
                    G.travel(i,j) = sqrt( (G.nodes{i}.x - G.nodes{j}.x)^2 + (G.nodes{i}.y - G.nodes{j}.y)^2 );
                end
            end
        end
        
        %% evaluate agent coordination
        function score = evaluate_coord(obj, a, b)
            score = 0;
            % go through nodes and see who has lower claim and assign
            % points for that claim
            for i=1:obj.n_nodes
                t_a = inf;
                r_a = 0;
                % go through A
                for j=1:length(a(:,1))
                    if a(j,1) == i
                        t_a = a(j,2);
                        r_a = a(j,3);
                    end
                end
                t_b = inf;
                r_b = 0;
                % go through B
                for j=1:length(b(:,1))
                    if b(j,1) == i
                        t_b = b(j,2);
                        r_b = b(j,3);
                    end
                end
                if t_a < t_b
                    score = score + r_a;
                else
                    score = score + r_b;
                end
            end
        end
    end
    
end


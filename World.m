classdef World
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        n_nodes
        nodes
        travel
        node_status
        dmcts_alpha
        dmcts_beta
        dmcts_gamma
        dmcts_min_sampling_threshold
        dmcts_max_rollout_depth
        dmcts_impact
        init_iters
        total_iters
        opt_iters
        dropout_rate
        com_mod
    end
    
    
    methods
        %% initilalize
        function G = World(n_nodes, t_iters)
            G.node_status = ones(1,n_nodes);
            G.n_nodes = n_nodes;
            G.total_iters = t_iters;
            for i=1:n_nodes
                n = Node(10*randn() - 5, 10*randn() -5, i);
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
        function score = evaluate_coord(obj, paths)
            score = 0;
            % go through nodes and see who has lower claim and assign
            % points for that claim
            for i=1:obj.n_nodes
                rewards = zeros(length(paths),1);
                times = inf * ones(length(paths),1);
                % go through each agent
                for ag = 1:length(paths)
                    path = paths{ag,:};
                    for j=1:length(path(:,1))
                        if path(j,1) == i
                            times(ag)= path(j,2);
                            rewards(ag) = path(j,3);
                        end
                    end
                end
                [~, mindex] = min(times);
                score = score + rewards(mindex);
            end
        end
        
        function G = get_test_params(G, test_var_name, test_var, test_iter) 
            if strcmp(test_var_name, 'alpha')
                G.dmcts_alpha = test_var(test_iter); % gradient descent on agent policy
            else
                G.dmcts_alpha = 0.1;
            end
            if strcmp(test_var_name,'beta')
                G.dmcts_beta = test_var(test_iter); % explore vs exploit
            else
                G.dmcts_beta = 1.41; % explore vs exploit
            end
            if strcmp(test_var_name,'gamma')
                G.dmcts_gamma = test_var(test_iter); % how long is my memory
            else
                G.dmcts_gamma = 0.99; % how long is my memory
            end
            if strcmp(test_var_name,'init iters')
                G.init_iters = test_var(test_iter); % how many iterations should I plan before broadcasting
                G.opt_iters = G.total_iters - G.init_iters;
            else
                G.init_iters = 0; % how many iterations should I plan before broadcasting
                G.opt_iters = G.total_iters - G.init_iters;
            end
            if strcmp(test_var_name,'dropout rate')
                G.dropout_rate = test_var(test_iter); % p com drops
            else
                G.dropout_rate = 0.05; % p com drops
            end
            if strcmp(test_var_name,'com mod')
                G.com_mod = test_var(test_iter); % how many do I transmit
            else
                G.com_mod = 1; % how many do I transmit
            end
            if strcmp(test_var_name,'min sampling')
                G.dmcts_min_sampling_threshold = test_var(test_iter); % how far down the tree do I sample
            else
                G.dmcts_min_sampling_threshold = 0.25; % how far down the tree do I sample
            end
            if strcmp(test_var_name,'rollout')
                G.dmcts_max_rollout_depth = test_var(test_iter); %rollout_test(test_iter);
            else
                G.dmcts_max_rollout_depth = 0; %rollout_test(test_iter);
            end
            if strcmp(test_var_name, 'impact')
                G.dmcts_impact = test_var(test_iter); % Do I use impact to get rewards or just raw reward?
            else
                G.dmcts_impact = 0;
            end
        end
        
    end
    
%% Setup test params

end




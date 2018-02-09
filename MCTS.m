classdef MCTS
    properties
        index % which node is it
        time % what time am I at
        my_reward % what am I worth
        down_branch_reward = 0; % how much are the line of my best children worth
        kids % my  kid nodes
        my_probability = 0 % how likely am I to be selected compared to other kids
        branch_probability = 0; % how likely is my branch to be selected
        depth % how deep in the tree am I?
        max_depth = 20; % how deep in the tree can I go?
        node_status % state of the world at m node
        n_pulls = 0; % how many times have I been pulled?
        max_rollouts = 3; % how many rollouts should I do?
        min_sampling_threshold = 0.001; % how deep in the tree do I sample?
        alpha = 0.01; % gradient descent rate
    end
    methods
        function  b = MCTS(ii, ti, di, nsi, md, G)
            b.index = ii;
            b.time = ti;
            b.my_reward = G.nodes{ii}.get_reward(ti);
            b.down_branch_reward = b.my_reward;
            b.depth = di;
            if di == 0
                b.my_probability = 1;
                b.branch_probability = 1;
            end
            b.node_status = nsi;
            b.node_status(ii)= 0;
            b.max_depth = md;
        end
        
        % plot tree distribution of each task
        function [times, probs] = plot_task_probs(obj, index, times, probs)
            if obj.index == index
                times(end+1) = obj.time;
                probs(end+1) = obj.branch_probability;
                return
            else
                for i=1:length(obj.kids)
                    if obj.kids{i}.branch_probability > obj.min_sampling_threshold
                        [times, probs] = obj.kids{i}.plot_task_probs(index, times, probs);
                    end
                end
            end
        end
        
        % make kids
        function obj = make_kids(obj, G)
            for i=1:length(G.nodes)
                if obj.node_status(i)
                    m = MCTS(i, obj.time + G.travel(obj.index, i), obj.depth+1, obj.node_status, obj.max_depth, G);
                    obj.kids{end+1} = m;
                end
            end
            obj = obj.sample_kids();
        end
        
        % greedy search tree
        function obj = greedy_search(obj, G)
            if obj.depth > obj.max_depth
                return
            end
            
            if isempty(obj.kids)
                obj = obj.make_kids(G);
            end
            
            if isempty(obj.kids)
                return;
            else
                maxI = -1;
                maxR = -1;
                for i=1:length(obj.kids)
                    if obj.kids{i}.my_reward > maxR
                        maxR = obj.kids{i}.my_reward;
                        maxI = i;
                    end
                end
                if maxI > -1
                    obj.kids{maxI} = obj.kids{maxI}.greedy_search(G);
                    obj.down_branch_reward = obj.kids{maxI}.down_branch_reward  + obj.my_reward;
                end
            end
        end
        
        % MCTS search tree
        function obj = mcts_search(obj, G)
            if obj.depth > obj.max_depth
                obj.down_branch_reward = obj.my_reward;
                return
            end
            obj.my_reward = G.nodes{obj.index}.get_reward(obj.time);
            if isempty(obj.kids)
                obj = obj.make_kids(G);
                %obj = obj.rollout_kids(G);
                if isempty(obj.kids)
                    obj.down_branch_reward = obj.my_reward;
                    return;
                end
            end
            
            [obj, maxI] = obj.ucb();    
            
            if maxI > -1
                obj.kids{maxI} = obj.kids{maxI}.mcts_search(G);
                if obj.kids{maxI}.down_branch_reward  + obj.my_reward > obj.down_branch_reward
                    obj.down_branch_reward = obj.kids{maxI}.down_branch_reward  + obj.my_reward;
                end
            end
            
        end
        
        % UCB
        function [obj, maxI] = ucb(obj)
            obj.n_pulls = obj.n_pulls + 1;
            minR = inf;
            maxR = -inf;
            for i=1:length(obj.kids)                    
                if obj.kids{i}.down_branch_reward < minR
                    minR = obj.kids{i}.down_branch_reward;
                end
                if obj.kids{i}.down_branch_reward > maxR
                    maxR = obj.kids{i}.down_branch_reward;
                end
            end

            if length(obj.kids) > 1
                maxM = -1;
                maxI = -1;
                for i=1:length(obj.kids)
                    rr = (obj.kids{i}.down_branch_reward-minR) / (maxR-minR);
                    ee = 1.41*sqrt(log(obj.n_pulls)/max(0.1,obj.kids{i}.n_pulls));
                    if rr + ee > maxM
                        maxM = rr + ee;
                        maxI = i;
                    end
                end
            else
                maxI = 1;
            end
        end
        
        % update my reward to account for team actions
        function obj = update_reward(obj, n)
            obj.my_reward = G.get_reward(obj.time) * (1-n.get_p_taken(obj.index, obj.time));
        end
        
        % updates probable actions
        function obj = sample_tree(obj)
            if isempty(obj.kids)
                return
            end
            
            maxR = -inf;
            maxI = -1;
            for i=1:length(obj.kids)
                if obj.kids{i}.down_branch_reward > maxR
                    maxR = obj.kids{i}.down_branch_reward;
                    maxI = i;
                end
            end
            
            obj.kids{maxI}.my_probability =  obj.kids{maxI}.my_probability*(1+obj.alpha);
            sumPP = 0.0;
            for i=1:length(obj.kids)
                sumPP = sumPP+ obj.kids{i}.my_probability;
            end
            
            for i=1:length(obj.kids)
               obj.kids{i}.my_probability = obj.kids{i}.my_probability/sumPP;  % normalize
               obj.kids{i}.branch_probability = obj.branch_probability * obj.kids{i}.my_probability;
               if obj.kids{i}.branch_probability > obj.min_sampling_threshold
                   obj.kids{i} = obj.kids{i}.sample_tree();
               end
            end
        end
        
        % rollout each kid
        function obj = rollout_kids(obj, G)
            for i=1:length(obj.kids)
                obj.kids{i}.rollout(1,G);
            end
        end
        
        % rollout mcts
        function obj = rollout(obj, r_depth, G)
            if obj.depth > obj.max_depth || r_depth > obj.max_rollouts
                obj.down_branch_reward = obj.my_reward;
                return
            end
            r_depth = r_depth + 1;
            
            if isempty(obj.kids)
                obj = obj.make_kids(G);
            end
            
            if isempty(obj.kids)
                obj.down_branch_reward = obj.my_reward;
                return;
            else
                maxI = -1;
                maxR = -1;
                for i=1:length(obj.kids)
                    if obj.kids{i}.my_reward > maxR
                        maxR = obj.kids{i}.my_reward;
                        maxI = i;
                    end
                end
                if maxI > -1
                    obj.kids{maxI} = obj.kids{maxI}.rollout(r_depth, G);
                    obj.down_branch_reward = obj.kids{maxI}.down_branch_reward  + obj.my_reward;
                end
            end
        end
        
        % creates initial estimate of probable actions
        function obj = sample_kids(obj)
            sumR = 0;
            for i=1:length(obj.kids)
                sumR = sumR + obj.kids{i}.down_branch_reward;
            end
            % minR / maxR does NOT work, puts all 0 to 1 individually,
            % allowing multiple to be 1.0
            for i=1:length(obj.kids)
                obj.kids{i}.my_probability = obj.kids{i}.down_branch_reward/sumR;
                obj.kids{i}.branch_probability = obj.branch_probability * obj.kids{i}.my_probability;
            end
        end
        
        % exploit tree
        function best_path = exploit(obj, best_path)
            best_path(length(best_path)+1) = obj.index;
            if obj.depth > obj.max_depth
                return
            end
            
            if isempty(obj.kids)
                return
            else
                maxI = -1;
                maxR = -1;
                for i=1:length(obj.kids)
                    if obj.kids{i}.down_branch_reward > maxR
                        maxR = obj.kids{i}.down_branch_reward;
                        maxI = i;
                    end
                end
                if maxI > -1
                    best_path = obj.kids{maxI}.exploit(best_path);
                else
                    return
                end
            end
        end
    end
    %%%%%%%%%%%%%%%%%% END MCTS CLASS %%%%%%%%%%%%%%%%%%
end


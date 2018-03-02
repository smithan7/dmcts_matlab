close all
clc

com_mod = [1,10,50,100,250,500,750,1000];
dropout_rate = [0.0, 0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]; % p com drops
min_sampling = [0.001,0.005,0.01,0.05,0.1,0.25,0.5, 0.75, 0.9, 0.95, 0.99, 1.0];
alpha_set = [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99,1.0];
beta_set = [0.141, 0.5, 1.0, 1.41, 2.0, 2.5, 5.0, 10.0];
gamma_set = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0]; 
rollout_set = [0,1,2,3,5,7,10];
init_iter_set = [0, 10, 50, 100, 250, 500, 1000];
impact_set = [0,1];
num_maps = 1;
n_nodes = 20;

test_var = init_iter_set;
test_var_name = 'init iters';
n_test_var = length(test_var);
total_iters = 5000;

big_coord_tracker = zeros(num_maps,n_test_var, total_iters);

for map_iter = 1:num_maps
    clearvars -except big_coord_tracker test_iter test_var num_maps total_iters map_iter test_var_name n_nodes n_test_var
    %for test_iter = 1:n_test_var
    percent_done = (map_iter-1)/num_maps;
    for test_iter = 1:1%n_test_var
        coord_temp = zeros(1,total_iters);
        a_temp = zeros(1,total_iters);
        b_temp = zeros(1,total_iters);
        rng(map_iter);
        G = World(n_nodes, total_iters);
        percent_done = (map_iter-1)/num_maps + (test_iter-1)/(num_maps*length(test_var));
        G = G.get_test_params(test_var_name, test_var, test_iter);
        
        time = 0;
        depth = 1;
        max_depth = n_nodes-1;%n_nodes/2 - 1;

        %%%%%%%%%%%%%%%%%% Dist-MCTS Test %%%%%%%%%%%%%%%%%%%%%%
        agents = {};
        agents{1} = Dist_MCTS(1, -1, time, depth, G.node_status, max_depth, G, 1);
        agents{2} = Dist_MCTS(n_nodes, -1, time, depth, G.node_status, max_depth, G, 2);
        agents{3} = Dist_MCTS(floor(n_nodes/2), -1, time, depth, G.node_status, max_depth, G, 3);
        
        team_coord = Coordinator(n_nodes);

        for iter=1:G.total_iters
          
            for a=1:length(agents)
                agents{a} = agents{a}.mcts_search(G, team_coord);
            end

            if iter >= G.init_iters     
                if rand() > G.dropout_rate && mod(iter,G.com_mod) == 0
                    for a=1:length(agents)
                        [agents{a}, team_coord] = agents{a}.sample_tree(team_coord);
                    end
                else
                    for a=1:length(agents)
                        [agents{a}, ~] = agents{a}.sample_tree(team_coord);
                    end
                end
            end


            paths = [];
            for a=1:length(agents)
                t_path = [];
                t_path = agents{a}.exploit(t_path);
                paths{a,1} = t_path;
            end
            
            if mod(iter,50) == 0
                figure(3)
                    hold off
                    for i=1:n_nodes
                        plot(G.nodes{i}.x, G.nodes{i}.y, 'rx')
                        hold on
                        txt = num2str(i);
                        text(G.nodes{i}.x+0.1, G.nodes{i}.y + 0.1, txt);
                    end
                    grid on
                    for ag=1:length(paths)
                        if ag == 1
                            sym = 'go';
                            co = 'g';
                        else
                            sym = 'bo';
                            co = 'b';
                        end
                        path = paths{ag};
                        for i=1:length(path(:,1))-1
                            plot([G.nodes{path(i,1)}.x, G.nodes{path(i+1,1)}.x], [G.nodes{path(i,1)}.y, G.nodes{path(i+1,1)}.y], co);
                        end
                        plot(G.nodes{path(1,1)}.x, G.nodes{path(1,1)}.y, sym)
                        plot(G.nodes{path(end,1)}.x, G.nodes{path(end,1)}.y, sym)
                    end
                title(num2str(iter))
                axis equal
                pause(0.1);
            end
            coord_temp(iter) = G.evaluate_coord(paths);
            
            for a=1:length(agents)    
                temp_rewards(a,iter) = agents{a}.down_branch_reward;
            end
            
            %a_dmcts_path = a_dmcts.exploit();
            %b_dmcts_path = b_dmcts.exploit()
            %coord_temp(i) = G.evaluate_coord(a_dmcts_path, b_dmcts_path);
            %a_temp(i) = a_dmcts.down_branch_reward;
            %b_temp(i) = b_dmcts.down_branch_reward;
        end
        big_coord_tracker(map_iter,test_iter,:) = coord_temp;
    end
end

coord_norm = zeros(num_maps, length(test_var), total_iters);
coord_mean = zeros(length(test_var), total_iters);
coord_std = zeros(length(test_var), total_iters);
max_reward = zeros(1,num_maps);
for  iter=1:length(test_var)
    for j=1:num_maps
        max_reward(j) = max(max(big_coord_tracker(j,:,1:end)));
        coord_norm(j,iter,:) = big_coord_tracker(j,iter,:) / 1;%max_reward(j);
    end
    coord_mean(iter,:) = mean(coord_norm(:,iter,:),1);
    if num_maps > 1
        coord_std(iter,:) = std(squeeze(coord_norm(:,iter,:)),1)/sqrt(num_maps);
    end
end
for  iter=1:length(test_var)
    figure(99)
        title(test_var_name);
        if num_maps == 1
            plot(coord_mean(iter,:), 'DisplayName', num2str(test_var(iter)));
        else
            errorbar(1:25:total_iters,coord_mean(iter,1:25:end), coord_std(iter,1:25:end), 'DisplayName', num2str(test_var(iter)));
        end
        hold on
        xlabel('Planning Iterations')
        ylabel('Collected Reward')
        %for j=1:num_maps
        %    plot(squeeze(coord_norm(j,i,:)), 'r');
        %end
        hold on
        grid on
end
legend('show')

for  iter=1:length(test_var)
     figure(100)
        title('A Down Branch');
        plot(1:total_iters,a_down_branch(iter,:),'DisplayName', num2str(test_var(iter)));
        hold on;
        grid on;
 end
legend('show')

for  iter=1:length(test_var)
      figure(101)
        title('B Down Branch');
        plot(1:total_iters,b_down_branch(iter,:),'DisplayName', num2str(test_var(iter)));
        hold on;
        grid on;
end
legend('show')
 
 for  iter=1:length(test_var)
      figure(102)
        title('A+B Down Branch');
        plot(1:total_iters,a_down_branch(iter,:) + b_down_branch(iter,:),'DisplayName', num2str(test_var(iter)));
        hold on;
        grid on;
end
legend('show')


% a_dmcts.down_branch_reward
% b_dmcts.down_branch_reward
% a_dmcts_path = [];
% a_dmcts_path = a_dmcts.exploit(a_dmcts_path);
% b_dmcts_path = [];
% b_dmcts_path = b_dmcts.exploit(b_dmcts_path);

figure(3)
    for iter=1:n_nodes
        plot(G.nodes{iter}.x, G.nodes{iter}.y, 'rx')
        hold on
        txt = num2str(iter);
        text(G.nodes{iter}.x+0.1, G.nodes{iter}.y + 0.1, txt);
    end
    grid on
    for iter=1:length(a_dmcts_path)-1
        plot([G.nodes{a_dmcts_path(iter,1)}.x, G.nodes{a_dmcts_path(iter+1,1)}.x], [G.nodes{a_dmcts_path(iter,1)}.y, G.nodes{a_dmcts_path(iter+1,1)}.y], 'g');
    end
    plot(G.nodes{a_dmcts_path(1,1)}.x, G.nodes{a_dmcts_path(1,1)}.y, 'go')
    plot(G.nodes{a_dmcts_path(end,1)}.x, G.nodes{a_dmcts_path(end,1)}.y, 'go')
    for iter=1:length(b_dmcts_path)-1
        plot([G.nodes{b_dmcts_path(iter,1)}.x, G.nodes{b_dmcts_path(iter+1,1)}.x], [G.nodes{b_dmcts_path(iter,1)}.y, G.nodes{b_dmcts_path(iter+1,1)}.y], 'b');
    end
    plot(G.nodes{b_dmcts_path(1,1)}.x, G.nodes{b_dmcts_path(1,1)}.y, 'bo')
    plot(G.nodes{b_dmcts_path(end,1)}.x, G.nodes{b_dmcts_path(end,1)}.y, 'bo')
    
%%%%%%%%%%%%%%%%%% Plots %%%%%%%%%%%%%%%%%%%%%%%%%%
    
%%

% for nn = 1:n_nodes
%     times = [0];
%     probs = [0];
%     [times, probs] = a_dmcts.plot_task_probs(nn,times,probs);
%     data = [times; probs];
%     s_data = sortrows(data',1)';
%     cum_prob = zeros(1, length(s_data(2,:)));
%     cum_prob(1) = s_data(2,1);
%     for i=2:length(s_data(2,:))
%         cum_prob(i) = cum_prob(i-1) + s_data(2,i);
%     end
%     
%     figure(4)
%         plot(s_data(1,:), cum_prob(:))
%         hold on
%         grid on
% end


        %%%%%%%%%%%%%%%%%%% Greedy Test %%%%%%%%%%%%%%%%%%
        % greedy_a = Greedy(index, time, depth, node_status, max_depth, G);
        % greedy_a = greedy_a.greedy_search(G);
        % greedy_coord = Coordinator(G.n_nodes);
        % for i=1:opt_iters
        %     [greedy_a, greedy_coord]  = greedy_a.sample_tree(greedy_coord);
        % end
        % 
        % greedy_a.down_branch_reward
        % greedy_path = [];
        % greedy_path = greedy_a.exploit(greedy_path)
        % figure(1)
        %     for i=1:n_nodes
        %         plot(G.nodes{i}.x, G.nodes{i}.y, 'rx')
        %         hold on
        %     end
        %     grid on
        %     for i=1:length(greedy_path)-1
        %         plot([G.nodes{greedy_path(i,1)}.x, G.nodes{greedy_path(i+1,1)}.x], [G.nodes{greedy_path(i,1)}.y, G.nodes{greedy_path(i+1,1)}.y], 'g');
        %     end
        %     plot(G.nodes{greedy_path(1,1)}.x, G.nodes{greedy_path(1,1)}.y, 'go')
        %     plot(G.nodes{greedy_path(end,1)}.x, G.nodes{greedy_path(end,1)}.y, 'go')

        %%%%%%%%%%%%%%%%%% MCTS Test %%%%%%%%%%%%%%%%%%%%%%
        % mcts_a = MCTS(index, time, depth, node_status, max_depth, G);
        % for i=1:opt_iters
        %     mcts_a = mcts_a.mcts_search(G);
        %     mcts_a = mcts_a.sample_tree();
        % end
        % 
        % mcts_a.down_branch_reward
        % mcts_path = [];
        % mcts_path = mcts_a.exploit(mcts_path);
        % figure(2)
        %     for i=1:n_nodes
        %         plot(G.nodes{i}.x, G.nodes{i}.y, 'rx')
        %         hold on
        %     end
        %     grid on
        %     for i=1:length(mcts_path)-1
        %         plot([G.nodes{mcts_path(i)}.x, G.nodes{mcts_path(i+1)}.x], [G.nodes{mcts_path(i)}.y, G.nodes{mcts_path(i+1)}.y], 'g');
        %     end
        %     plot(G.nodes{mcts_path(1)}.x, G.nodes{mcts_path(1)}.y, 'go')
        %     plot(G.nodes{mcts_path(end)}.x, G.nodes{mcts_path(end)}.y, 'go')

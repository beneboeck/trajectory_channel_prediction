addpath('.\quadriga_src')
close all
clear all
clc

%% (A) global variables

no_runs = 500;                                     % Number of total simulations
no_trajectories = 100;                             % Number of simulated trajectories per simulation
centerFrequency = 2.1e9;                           % Center frequency
bandwidth = 1e6;                                    % Bandwidth in Hz
lambda = 3e8/centerFrequency;                       % Corresponding wavelength
antennaHeight = 10;                                 % Antenna height of the bs station in [m]
antennaSpacing = 1/2;                               % Antenna spacing in multiples of the wave length
noV = 32;                                            % Number of vertical antenna elements
noH = 1;                                            % Number of horizontal antenn elements
noAnBS = noV*noH;                                   % Total number of antennas at the BS

%%% Parameters
minRadius = 50;                                     % Minium distance of the trajectory's initial position to the BS (2D)
maxRadius = 150;                                     % Maximum distance of the trajectory's initial position to the BS (2D)
angleSpread = 45;                                   % Maximum angle spread between the trajectories' initial positions (2D)
mtHeight = 1.5;                                     % Height of the MTs
snapshotDensity = 0.5e-3;                           % Snapshot density in [s]
durationTrajectory = 7.5e-3;                         % Duration of one Trajectory

H_real = zeros(no_runs * no_trajectories,noAnBS,durationTrajectory/snapshotDensity+1);
H_imag = zeros(no_runs * no_trajectories,noAnBS,durationTrajectory/snapshotDensity+1);
path_gains_total = zeros(no_runs * no_trajectories,durationTrajectory/snapshotDensity+1);

for o = [1:no_runs]
    disp('run')
    o

    %velocity computation (velocity is drawn from a rayleigh distribution as
    %abs(complexGauss(0,sig = 2)) (mean somewhere around 2.5m/s and at most 8m/s)
    randx = 2 * (randn(no_trajectories,1) + 1j * randn(no_trajectories,1));
    mtVelocity = abs(randx);                          % Velocity of the MTs in [m]/[s]
    %mtVelocity = 3 * ones(no_trajectories,1)

    lengthTrajectory = mtVelocity .* durationTrajectory; % Length of the Trajectories
    spatialDensity = lengthTrajectory ./ (durationTrajectory/snapshotDensity);       % Snapshot density in [m]
    noSnapshots = durationTrajectory/snapshotDensity + 1;                            % Number of Snapshots
    %figure;
    %hist(mtVelocity,200)
    %set(gca,'FontSize',18)
    %xlabel('$v [m/s]$','Interpreter','Latex','FontSize',25)

    %% (B) simulation parameters

    s = qd_simulation_parameters;                           % Set up simulation parameters
    s.show_progress_bars = 0;                               % Disable progress bars
    s.center_frequency = centerFrequency;                   % Set center frequency
    s.sample_density = 2;                                   % 2 samples per half-wavelength
    s.use_absolute_delays = 1;                              % Include delay of the LOS path

    %% (C) layout

    l = qd_layout(s);                                       % Create new QuaDRiGa layout

    % C.1) base station

    l.no_tx = 1;
    l.tx_position(3) = antennaHeight;
    l.tx_array = qd_arrayant('3gpp-3d', noV, noH, centerFrequency, 1, 0, antennaSpacing);


    %%%%%%%%%%% question %%%%%%%%%%%%%%
    % why do we place the BS antennas like that? Sure, we ensure that the
    % antennas are lambda/2 seperated but there is some odd offset in the
    % global placement
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for n=1:noV
        for nn=1:noH
            indeces = (n-1)*noH+nn;
            l.tx_array.element_position(1,indeces) =  (nn)*antennaSpacing*lambda  - lambda/4 - noV/2*antennaSpacing*lambda;
            l.tx_array.element_position(2,indeces) = 0;
            l.tx_array.element_position(3,indeces) = (n)*antennaSpacing*lambda - lambda/4 - noH/2*antennaSpacing*lambda + antennaHeight;
        end
    end


    % C.2) mobile terminals

    %
    %figure_generation
    pos = zeros(no_trajectories,noSnapshots,2);
    %
    mtAntenna = qd_arrayant('omni');
    l.no_rx = no_trajectories;

    mtRadius = (maxRadius-minRadius) * rand(no_trajectories,1) + minRadius;
    mtPhase = angleSpread * rand(no_trajectories,1) - angleSpread/2;

    initial_positions = zeros(no_trajectories,3);

    for i = 1:no_trajectories
        if mod(i,1000) == 0
            disp('step')
            i
        end
        l.rx_array(i) = mtAntenna.copy;
        l.rx_track(i) = qd_track('linear', lengthTrajectory(i), rand(1) * 2*pi);
        l.rx_track(i).name = strcat('track',int2str(i));
        l.rx_track(i).interpolate('distance',spatialDensity(i));
        l.rx_track(i).initial_position = [mtRadius(i) * cos(mtPhase(i)*(2*pi)/(360));mtRadius(i) * sin(mtPhase(i)*(2*pi)/(360));mtHeight];

        pos(i,:,1) = mtRadius(i) * cos(mtPhase(i)*(2*pi)/(360)) + l.rx_track(i).positions(1,:);
        pos(i,:,2) = mtRadius(i) * sin(mtPhase(i)*(2*pi)/(360)) + l.rx_track(i).positions(2,:);

        initial_positions(i,1) = mtRadius(i) * cos(mtPhase(i)*(2*pi)/(360));
        initial_positions(i,2) = mtRadius(i) * sin(mtPhase(i)*(2*pi)/(360));
        initial_positions(i,3) = mtHeight;
        l.rx_track(i).scenario{1} = '3GPP_38.901_UMi_NLOS';
    end

%ab = l.rx_track(3).positions;

%start_pos = initial_positions(:,1:2);


%%% testing
%a = zeros(no_trajectories,2);
%for i = 1:no_trajectories
%    a(i,:) = l.rx_track(i).initial_position(1:2);
%end
%figure;
%scatter(a(:,1),a(:,2))

%l.visualize([],[],0);
%hold on

%[map,x_coords,y_coords] = l.power_map('3GPP_3D_UMi_NLOS','quick',0.5,-10,160,-125,125,0);
%xlim([-5,170]);
%ylim([-50,50]);
%P = 10*log10( sum(cat(3,map{:}),3));
%P = sum(P,4);
%imagesc( x_coords, y_coords, P );
%legend('Location','northeast')
%figure;
%scatter(l.tx_array.element_position(1,:),l.tx_array.element_position(3,:));
%xlim([-10,125]);
%ylim([-60,60]);
%figure;
%set(gcf, 'Units', 'normalized', 'Position', [0.2, 0.1, 0.7, 0.7]); 
%set(groot,'defaultAxesTickLabelInterpreter','latex');  
%set(groot,'defaulttextinterpreter','latex');
%set(groot,'defaultLegendInterpreter','latex');
%imagesc(x_coords,y_coords,P);
%hold on;
%scatter([0],[0],350,'MarkerEdgeColor','red','LineWidth',3,'DisplayName','BS position','Marker','square')
%hold on;
%for i = 1:no_trajectories
%    plot(pos(i,:,1),pos(i,:,2),'LineWidth',1.8,'Color','blue','DisplayName','trajectories');
%    scatter(start_pos(i,1),start_pos(i,2),100,'Marker','square','LineWidth',2,'MarkerEdgeColor','blue','DisplayName','UE starting positions')
%    hold on;
%end

%angles = linspace(-angleSpread/2,angleSpread/2,100);
%radiuse = linspace(minRadius,maxRadius,100);

%line1 = [minRadius * cos(angles*(2*pi)/(360));minRadius * sin(angles*(2*pi)/(360))];
%line2 = [maxRadius * cos(angles*(2*pi)/(360));maxRadius * sin(angles*(2*pi)/(360))];
%line3 = [radiuse * cos(-angleSpread/2*(2*pi)/(360));radiuse * sin(-angleSpread/2*(2*pi)/(360))];
%line4 = [radiuse * cos(angleSpread/2*(2*pi)/(360));radiuse * sin(angleSpread/2*(2*pi)/(360))];

%plot(line1(1,:),line1(2,:),'Color','black','LineWidth',2.5,'LineStyle','--','DisplayName','considered area');
%hold on;
%plot(line2(1,:),line2(2,:),'Color','black','LineWidth',2.5,'LineStyle','--');
%hold on;
%plot(line3(1,:),line3(2,:),'Color','black','LineWidth',2.5,'LineStyle','--');
%hold on;
%plot(line4(1,:),line4(2,:),'Color','black','LineWidth',2.5,'LineStyle','--');
%hold on;
%legend(legendUnq(),'Location','northwest','FontSize',23,'Interpreter','latex');
%xlim([-2.5,105]);
%ylim([-45,45]);
%ax = gca; 
%ax.FontSize = 26;
%xlabel('$$x$$ [m]','FontSize',29);
%ylabel('$$y$$ [m]','FontSize',29)
%%%%%%%


%% (D) channel coefficients

    C = l.get_channels;

    H = zeros(no_trajectories,noAnBS,noSnapshots);
    path_gains = zeros(no_trajectories,noSnapshots);
    for i = 1:no_trajectories 
        if mod(i,100) == 0
            disp('step2')
            i
        end
        H(i,:,:) = squeeze(C(i).fr(bandwidth,1));
        for j = 1:noSnapshots
            path_gains(i,j) = C(i).par.pg(j);
        end
    end
    % comment: additional saves: PG_normalization_factor + initial_positions
    %figure;
    %a = squeeze(real(H(1,:,:)));
    %a = a./max(a) * 256;
    %image(a)
    %set(gca,'FontSize',18)
    %title('3m/s, 0.5ms snapshots, 15ms trajectory','FontSize',18)
    %xlabel('Snapshots','FontSize',18)
    %ylabel('BS antennas','FontSize',18)

    %% (E) textfile 

    H_real((o-1) * no_trajectories + 1: o * no_trajectories,:,:) = real(H);
    H_imag((o-1) * no_trajectories + 1: o * no_trajectories,:,:) = imag(H);
    path_gains_total((o-1) * no_trajectories + 1: o * no_trajectories,:) = path_gains;
end


H_real_train = H_real(1:0.8 * no_runs * no_trajectories,:,:);
H_real_val = H_real(0.8 * no_runs * no_trajectories + 1:0.9 * no_runs * no_trajectories,:,:);
H_real_test = H_real(0.9 * no_runs * no_trajectories + 1:end,:,:);

H_imag_train = H_imag(1:0.8 * no_runs * no_trajectories,:,:);
H_imag_val = H_imag(0.8 * no_runs * no_trajectories + 1:0.9 * no_runs * no_trajectories,:,:);
H_imag_test = H_imag(0.9 * no_runs * no_trajectories + 1:end,:,:);

path_gains_train = path_gains_total(1:0.8 * no_runs * no_trajectories,:);
path_gains_val = path_gains_total(0.8 * no_runs * no_trajectories + 1:0.9 * no_runs * no_trajectories,:);
path_gains_test = path_gains_total(0.9 * no_runs * no_trajectories + 1:end,:);



save('../Simulations//trajectory_channel_prediction/data/H_real_train500_100.mat','H_real_train','-v7.3');
save('../Simulations//trajectory_channel_prediction/data/H_real_val500_100.mat','H_real_val','-v7.3');
save('../Simulations//trajectory_channel_prediction/data/H_real_test500_100.mat','H_real_test','-v7.3');

save('../Simulations//trajectory_channel_prediction/data/H_imag_train500_100.mat','H_imag_train','-v7.3');
save('../Simulations//trajectory_channel_prediction/data/H_imag_val500_100.mat','H_imag_val','-v7.3');
save('../Simulations//trajectory_channel_prediction/data/H_imag_test500_100.mat','H_imag_test','-v7.3');

save('../Simulations//trajectory_channel_prediction/data/path_gains_train500_100.mat','path_gains_train','-v7.3');
save('../Simulations//trajectory_channel_prediction/data/path_gains_val500_100.mat','path_gains_val','-v7.3');
save('../Simulations//trajectory_channel_prediction/data/path_gains_test500_100.mat','path_gains_test','-v7.3');



%save('../data/ULA_R1_H_imag_DRESDEN_NLOS_5000_trajectories_4_test.mat','H_imag','-v7.3');
%save('../data/ULA_R1_initial_positions_DRESDEN_NLOS_5000_trajectories_4_test.mat','initial_positions','-v7.3')
%save('../data/ULA_R1_path_gains_DRESDEN_NLOS_5000_trajectories_4_test.mat','path_gains','-v7.3')

%save('../data/ULA_R1_H_real_DRESDEN_NLOS_5000_trajectories_4_double.mat','H_real','-v7.3');
%save('../data/ULA_R1_H_imag_DRESDEN_NLOS_5000_trajectories_4_double.mat','H_imag','-v7.3');
%save('../data/ULA_R1_initial_positions_DRESDEN_NLOS_5000_trajectories_4_double.mat','initial_positions','-v7.3')
%save('../data/ULA_R1_path_gains_DRESDEN_NLOS_5000_trajectories_4_double.mat','path_gains','-v7.3')


%writematrix(real(H),'test_data_5_MTs_10100_trajectories_4_double.mat','path_gains','-v7.3')


%writematrix(real(H),'test_data_5_MTs_101_samples_32_channels_real.txt')
%writematrix(imag(H),'test_data_5_MTs_101_samples_32_channels_imag.txt')
% alternative: save as .mat ,_file with optional argument  h5py python
%save('H_fr.mat', 'H_fr', '-v7.3');
% package


%%%%%%%%%%%%%%% question %%%%%%%%%%%%%
% why is F_a and F_b of every antenna element the same whereas they are
% placed at different locations?
% -> maybe understandable because F_a and F_b are meant locally, but why is
% then the power_map not shifted somehow -> look at antenna_positions.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%[map,x_coords,y_coords] = l.power_map('3GPP_38.901_UMa_LOS','quick',0.01,-3,3,-3,3,0);
%P = 10*log10( sum(cat(3,map{:}),3));
%P = P(:,:,:,10);
%figure;
%imagesc( x_coords, y_coords, P );
%samples_32_channels_real.txt')
%writematrix(imag(H),'test_data_5_MTs_101_samples_32_channels_imag.txt')
% alternative: save as .mat ,_file with optional argument  h5py python
%save('H_fr.mat', 'H_fr', '-v7.3');
% package


%%%%%%%%%%%%%%% question %%%%%%%%%%%%%
% why is F_a and F_b of every antenna element the same whereas they are
% placed at different locations?
% -> maybe understandable because F_a and F_b are meant locally, but why is
% then the power_map not shifted somehow -> look at antenna_positions.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%[map,x_coords,y_coords] = l.power_map('3GPP_38.901_UMa_LOS','quick',0.01,-3,3,-3,3,0);
%P = 10*log10( sum(cat(3,map{:}),3));
%P = P(:,:,:,10);
%figure;
%imagesc( x_coords, y_coords, P );

#include "my_dqn.h"
#include "environment.h"
#include <ale_interface.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <cmath>
#include <iostream>
#include <string>
#include <deque>
#include <algorithm>

#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

DEFINE_int32(verbose, 0, "verbose output");
DEFINE_int32(gpu, 0, "Use GPU to brew Caffe");
DEFINE_int32(gui, 0, "Open a GUI window");
DEFINE_string(rom, "pong.bin", "Atari 2600 ROM to play");
DEFINE_string(solver, "my_dqn_solver.prototxt", "Solver parameter file (*.prototxt)");
DEFINE_string(snapshot, "", "Snapshot to resume training (*.solverstate). WARNING : memory buffer is not saved.");
DEFINE_int32(memory, 500000, "Capacity of replay memory");
DEFINE_int32(clone_frequency, 10000, "How often (steps) the target net is updated");
DEFINE_int32(explore, 1000000, "Number of iterations needed for epsilon to reach 0.1");
DEFINE_double(gamma, 0.95, "Discount factor of future rewards (0,1]");
DEFINE_int32(memory_threshold, 10000, "Enough amount of transitions to start learning");
DEFINE_int32(skip_frame, 4, "Number of frames skipped. 1 for desactivating frame skipping");
DEFINE_string(model, "pong_iter_8400000.caffemodel", "Model file to load");
DEFINE_int32(evaluate, 0, "Evaluation mode: only playing a game, no updates");
DEFINE_double(evaluate_with_epsilon, 0.05, "Epsilon value to be used in evaluation mode");
DEFINE_double(repeat_games, 1, "Number of games played in evaluation mode");
DEFINE_int32(steps_per_epoch, 5000, "Number of training steps per epoch");

double CalculateEpsilon(const int iter)
{
	if (iter < FLAGS_explore)
	{
		return 1.0 - 0.9 * (static_cast<double>(iter) / FLAGS_explore);
	}
	else
	{
		return 0.1;
	}
}

void SaveScreen(my_dqn::EnvironmentSp environmentSp, std::string filename)
{
	my_dqn::FrameDataSp frame = environmentSp->PreprocessScreen();

	cv::imwrite(filename, cv::Mat(my_dqn::kCroppedFrameSize, my_dqn::kCroppedFrameSize, CV_8UC1, frame->data()));
}

/**
 * Play one episode and return the total score
 */
double PlayOneEpisode(my_dqn::EnvironmentSp environmentSp, my_dqn::My_DQN* dqn, const double epsilon, const bool update)
{
	assert(!environmentSp->EpisodeOver());

	std::deque<my_dqn::FrameDataSp> past_frames;
	double total_score = 0.0;
	
	while (!environmentSp->EpisodeOver())
	{
		double reward = 0.0;
		int currentAction = 0;
		my_dqn::State input_frames;

		const my_dqn::FrameDataSp current_frame = environmentSp->PreprocessScreen();
		past_frames.push_back(current_frame);

		if (past_frames.size() < my_dqn::kInputFrameCount)
		{
			// If there are not past frames enough for DQN input, just select NOOP
			for (int i = 0; i < FLAGS_skip_frame; ++i)
			{
				reward += environmentSp->Act(currentAction);
			}

			total_score += reward;
		}
		else
		{
			if (past_frames.size() > my_dqn::kInputFrameCount)
			{
				past_frames.pop_front();
			}

			std::copy(past_frames.begin(), past_frames.end(), input_frames.begin());

			//Select action
			currentAction = dqn->SelectAction(input_frames, epsilon);

			//repeat action
			for (int i = 0; i < FLAGS_skip_frame; ++i)
			{
				reward += environmentSp->Act(currentAction);
			}

			total_score += reward;

			// Rewards for DQN are normalized as follows:
			// 1 for any positive score, -1 for any negative score, otherwise 0
			reward = reward == 0 ? 0 : reward /= std::abs(reward);

			if (update)
			{
				// Add the current image to replay memory				
				dqn->AddImageToMemory(my_dqn::MemoryImage(currentAction, current_frame, reward));

				// If the episode is over, add a null image to the replay memory
				if (environmentSp->EpisodeOver())
				{
					dqn->AddImageToMemory(my_dqn::MemoryImage(currentAction, nullptr, reward));
				}

				// If the size of replay memory is enough, update DQN
				if (dqn->memory_size() > FLAGS_memory_threshold)
				{
					dqn->Update();
				}
			}
		}
	}
	environmentSp->Reset();
	return total_score;
}

int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	google::InitGoogleLogging(argv[0]);
	google::LogToStderr();

	if (FLAGS_gpu)
	{
		caffe::Caffe::set_mode(caffe::Caffe::GPU);
	}
	else
	{
		caffe::Caffe::set_mode(caffe::Caffe::CPU);
	}

	my_dqn::EnvironmentSp environmentSp = my_dqn::CreateEnvironment(FLAGS_gui, FLAGS_rom);

	// Get the vector of legal actions
	const my_dqn::Environment::ActionVec legal_actions = environmentSp->GetMinimalActionSet();

	my_dqn::My_DQN dqn(environmentSp, legal_actions, FLAGS_solver, FLAGS_snapshot, FLAGS_memory, FLAGS_clone_frequency, FLAGS_gamma, FLAGS_verbose);

	dqn.Initialize();

	if (!FLAGS_model.empty())
	{
		// Just load the given model
		LOG(INFO) << "Loading " << FLAGS_model;
		dqn.LoadTrainedModel(FLAGS_model);
	}

	if (FLAGS_evaluate)
	{
		double total_score = 0.0;

		for (int i = 0; i < FLAGS_repeat_games; ++i)
		{
			LOG(INFO) << "game: ";
			const double score = PlayOneEpisode(environmentSp, &dqn, FLAGS_evaluate_with_epsilon, false);
			LOG(INFO) << "score: " << score;
			total_score += score;
		}

		LOG(INFO) << "Mean score: " << total_score / FLAGS_repeat_games;

		return 0;
	}

	double total_score = 0.0;
	double epoch_total_score = 0.0;
	int epoch_episode_count = 0.0;
	double total_time = 0.0;
	int next_epoch_boundary = FLAGS_steps_per_epoch;
	double running_average = 0.0;
	double plot_average_discount = 0.05;

	std::ofstream training_data(".//training_log.csv");

	training_data << FLAGS_rom << ";" << FLAGS_steps_per_epoch << ";;;" << std::endl;
	training_data << "Epoch;Epoch avg score;Hours training;Number of episodes;episodes in epoch" << std::endl;

	caffe::SolverParameter solver_param;
	caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);

	for (int episode = 0; dqn.current_iteration() < solver_param.max_iter(); episode++)
	{
		std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

		epoch_episode_count++;
		const double epsilon = CalculateEpsilon(dqn.current_iteration());
		double train_score = PlayOneEpisode(environmentSp, &dqn, epsilon, true);

		epoch_total_score += train_score;

		if (dqn.current_iteration() > 0)  // started training?
		{
			std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
			total_time += std::chrono::duration<double, std::milli>(end - start).count();
		}

		LOG(INFO) << "training score(" << episode << "): " << train_score << std::endl;
		LOG(INFO) << "Memory size: " << dqn.memory_size() << " (" << dqn.memory_size() * (sizeof(int) + sizeof(float) + sizeof(my_dqn::FrameData)) << " bytes)" << std::endl;

		if (episode == 0)
		{
			running_average = train_score;
		}
		else
		{
			running_average = train_score*plot_average_discount + running_average*(1.0 - plot_average_discount);
		}

		if (dqn.current_iteration() >= next_epoch_boundary)
		{
			double hours = total_time / 1000. / 3600.;

			int epoc_number = static_cast<int>((next_epoch_boundary) / FLAGS_steps_per_epoch);

			LOG(INFO) << "epoch(" << epoc_number << ":" << dqn.current_iteration() << "): " << "average score " << running_average << " in " << hours << " hour(s)";

			if (dqn.current_iteration())
			{
				double hours_for_million = hours / (dqn.current_iteration() / 1000000.0);
				LOG(INFO) << "Estimated Time for 1 million iterations: " << hours_for_million << " hours";
			}

			training_data << epoc_number << ";" << running_average << ";" << hours << ";" << episode << ";" << epoch_episode_count << std::endl;

			epoch_total_score = 0.0;
			epoch_episode_count = 0;

			while (next_epoch_boundary < dqn.current_iteration())
			{
				next_epoch_boundary += FLAGS_steps_per_epoch;
			}
		}
	}

	training_data.close();
}


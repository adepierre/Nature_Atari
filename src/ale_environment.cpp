#include "environment.h"
#include <ale_interface.hpp>
#include <glog/logging.h>
#include <iostream>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace my_dqn
{

	class ALEEnvironment : public Environment
	{

	public:
		ALEEnvironment(bool gui, const std::string rom_path) : ale_(gui)
		{
			ale_.setBool("display_screen", gui);
			ale_.setFloat("repeat_action_probability", 0.00);
			ale_.loadROM(rom_path);

			ActionVect av = ale_.getMinimalActionSet();
			for (int i = 0; i < av.size(); i++)
			{
				legal_actions_.push_back(static_cast<ActionCode>(av[i]));
			}
		}

		FrameDataSp PreprocessScreen()
		{
			std::vector<uint8_t> screen;
			ale_.getScreenGrayscale(screen);

			cv::Mat greyScreen = cv::Mat(ale_.getScreen().height(), ale_.getScreen().width(), CV_8UC1, screen.data());

			FrameDataSp fdataSp = std::make_shared<FrameData>();
			cv::Mat resizedScreen = cv::Mat(kCroppedFrameSize, kCroppedFrameSize, CV_8UC1, fdataSp->data());
			cv::resize(greyScreen, resizedScreen, cv::Size(kCroppedFrameSize, kCroppedFrameSize), 0.0, 0.0, cv::InterpolationFlags::INTER_CUBIC);
			return fdataSp;
		}

		double ActNoop() 
		{
			double reward = 0;
			if(!ale_.game_over())
			{
				reward += ale_.act(PLAYER_A_NOOP);
			}
			return reward;
		}

		double Act(int action) 
		{
			double reward = 0;
			if(!ale_.game_over()) 
			{
				reward += ale_.act((Action)action);
			}
			return reward;
		}

		void Reset() 
		{
			ale_.reset_game();
		}

		bool EpisodeOver() 
		{
			return ale_.game_over();
		}

		std::string action_to_string(Environment::ActionCode a)
		{
			return action_to_string(static_cast<Action>(a));
		}

		const ActionVec& GetMinimalActionSet()
		{
			return legal_actions_;
		}

	private:

		ALEInterface ale_;
		ActionVec legal_actions_;

	};

	EnvironmentSp CreateEnvironment(bool gui, const std::string rom_path) 
	{
		return std::make_shared<ALEEnvironment>(gui, rom_path);
	}

}  // namespace fast_dqn
#ifndef SRC_FAST_DQN_H_
#define SRC_FAST_DQN_H_

#include "environment.h"
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <memory>
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <utility>
#include <deque>
#include <string>
#include <chrono>

namespace my_dqn
{

	const int kRawFrameHeight = 250;
	const int kRawFrameWidth = 160;
	const int kCroppedFrameSize = 84;
	const int kCroppedFrameDataSize = kCroppedFrameSize * kCroppedFrameSize;
	const int kInputFrameCount = 4;
	const int kInputDataSize = kCroppedFrameDataSize * kInputFrameCount;
	const int kMinibatchSize = 32;
	const int kMinibatchDataSize = kInputDataSize * kMinibatchSize;
	const float kGamma = 0.95f;
	const int kOutputCount = 18;

	const std::string frames_layer_name = "frames_input_layer";
	const std::string target_layer_name = "target_input_layer";
	const std::string filter_layer_name = "filter_input_layer";

	const std::string train_frames_blob_name = "frames";
	const std::string test_frames_blob_name = "all_frames";
	const std::string target_blob_name = "target";
	const std::string filter_blob_name = "filter";
	const std::string q_values_blob_name = "q_values";

	using FrameData = std::array<uint8_t, kCroppedFrameDataSize>;
	using FrameDataSp = std::shared_ptr<FrameData>;
	using State = std::array<FrameDataSp, kInputFrameCount>;
	using InputStateBatch = std::vector<State>;


	using FramesLayerInputData = std::array<float, kMinibatchDataSize>;
	using TargetLayerInputData = std::array<float, kMinibatchSize * kOutputCount>;
	using FilterLayerInputData = std::array<float, kMinibatchSize * kOutputCount>;


	typedef struct ActionValue
	{
		ActionValue(const Environment::ActionCode _action, const float _q_value) :
			action(_action), q_value(_q_value)
		{

		}
		const Environment::ActionCode action;
		const float q_value;
	} ActionValue;

	typedef struct MemoryImage
	{
		MemoryImage(const Environment::ActionCode action_, const FrameDataSp image_, const float reward_) :
			action(action_), image(image_), reward(reward_)
		{

		}
		const Environment::ActionCode action;
		const FrameDataSp image;
		const float reward;
	} MemoryImage;


	/**
	  * Transition
	  */
	class Transition
	{
	public:

		Transition(const State state, Environment::ActionCode action,
			double reward, FrameDataSp next_frame) :
			state_(state),
			action_(action),
			reward_(reward),
			next_frame_(next_frame)
		{

		}

		bool is_terminal() const { return next_frame_ == nullptr; }

		const State GetNextState() const;

		const State& GetState() const { return state_; }

		Environment::ActionCode GetAction() const { return action_; }

		double GetReward() const { return reward_; }

	private:
		const State state_;
		Environment::ActionCode action_;
		double reward_;
		FrameDataSp next_frame_;
	};
	typedef std::shared_ptr<Transition> TransitionSp;

	/**
	 * Deep Q-Network
	 */
	class My_DQN
	{
	public:
		My_DQN(
			EnvironmentSp environmentSp,
			const Environment::ActionVec& legal_actions,
			const std::string& solver_param,
			const std::string &snapshot_file,
			const int replay_memory_capacity,
			const int clone_target_frequency,
			const double gamma,
			const bool verbose) :
			environmentSp_(environmentSp),
			legal_actions_(legal_actions),
			solver_param_(solver_param),
			snapshot_file_(snapshot_file),
			replay_memory_capacity_(replay_memory_capacity),
			gamma_(gamma),
			verbose_(verbose),
			random_engine_(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
			clone_frequency_(clone_target_frequency),
			last_clone_iter_(0) {   // Iteration in which the net was last cloned
		}

		/**
		 * Initialize DQN. Must be called before calling any other method.
		 */
		void Initialize();

		/**
		 * Load a trained model from a file.
		 */
		void LoadTrainedModel(const std::string& model_file);

		/**
		 * Select an action by epsilon-greedy.
		 */
		Environment::ActionCode SelectAction(const State& input_frames, double epsilon);

		/**
		 * Add an image to replay memory
		 */
		void AddImageToMemory(const MemoryImage& memoryImage_);

		/**
		 * Update DQN using one minibatch
		 */
		void Update();

		/**
		* Return replay buffer size
		*/
		int memory_size() const { return replay_memory_.size(); }

		/**
		 * Copy the current training net_ to the target_net_
		 */
		void CloneTrainingNetToTargetNet() { CloneNet(net_); }

		/**
		 * Return the current iteration of the solver
		 */
		int current_iteration() const { return solver_->iter(); }

	private:
		using SolverSp = std::shared_ptr<caffe::Solver<float>>;
		using NetSp = boost::shared_ptr<caffe::Net<float>>;
		using BlobSp = boost::shared_ptr<caffe::Blob<float>>;
		using MemoryDataLayerSp = boost::shared_ptr<caffe::MemoryDataLayer<float>>;


		Environment::ActionVec SelectActions(const InputStateBatch& frames_batch, const double epsilon);
		ActionValue SelectActionGreedily(NetSp net, const State& last_frames);
		std::vector<ActionValue> SelectActionGreedily(NetSp, const InputStateBatch& last_frames);

		/**
		  * Clone the given net and store the result in clone_net_
		  */
		void CloneNet(NetSp net);

		/**
		 * Init the target and filter layers.
		 */
		void InitNet(NetSp net);

		/**
		  * Input data into the Frames/Target/Filter layers of the given
		  * net. This must be done before forward is called.
		  */
		void InputDataIntoLayers(NetSp net,
			const FramesLayerInputData& frames_data,
			const TargetLayerInputData& target_data,
			const FilterLayerInputData& filter_data);

		EnvironmentSp environmentSp_;
		const Environment::ActionVec legal_actions_;
		const int replay_memory_capacity_;
		const double gamma_;
		std::deque<MemoryImage> replay_memory_;
		TargetLayerInputData dummy_input_data_;

		const std::string solver_param_;
		const std::string snapshot_file_;
		SolverSp solver_;
		NetSp net_; // The primary network used for action selection.
		NetSp target_net_; // Clone used to generate targets.
		const int clone_frequency_; // How often (steps) the target_net is updated
		int last_clone_iter_; // Iteration in which the net was last cloned


		std::mt19937 random_engine_;
		bool verbose_;
	};


}  // namespace fast_dqn

#endif  // SRC_FAST_DQN_H_

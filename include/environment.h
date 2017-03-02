#ifndef SRC_ENVIRONMENT_H_
#define SRC_ENVIRONMENT_H_
#include <vector>
#include <memory>
#include <array>

namespace my_dqn 
{

  // Abstract environment class
  // implementation must define the class 
  //    EnvironmentSp CreateEnvironment( bool gui, const std::string rom_path);

class Environment;
typedef std::shared_ptr<Environment> EnvironmentSp;

class Environment 
{
 public:
  typedef std::vector<int> ActionVec;
  typedef int ActionCode;

  static const int kRawFrameHeight = 250;
  static const int kRawFrameWidth = 160;
  static const int kCroppedFrameSize = 84;
  static const int kCroppedFrameDataSize =
    kCroppedFrameSize * kCroppedFrameSize;
  static const int kInputFrameCount = 4;
  static const int kInputDataSize =
    kCroppedFrameDataSize * kInputFrameCount;

  using FrameData = std::array<uint8_t, kCroppedFrameDataSize>;
  using FrameDataSp = std::shared_ptr<FrameData>;
  using State = std::array<FrameDataSp, kInputFrameCount>;

  virtual FrameDataSp PreprocessScreen() = 0;

  virtual double ActNoop() = 0;

  virtual double Act(int action) = 0;

  virtual void Reset() = 0;

  virtual bool EpisodeOver() = 0;

  virtual std::string action_to_string(ActionCode a) = 0;

  virtual const ActionVec& GetMinimalActionSet() = 0;

};

// Factory method
EnvironmentSp CreateEnvironment(bool gui, const std::string rom_path);

}  // namespace fast_dqn
#endif  // SRC_ENVIRONMENT_H_
# Gym implementation for Xarm 

OpenAI Gym Xarm7 robot environment implemented with PyBullet.

## Installation

```shell
git clone https://github.com/jc-bao/gym-xarm.git
cd gym-xarm
pip install -e .
```

## Use Cases

### Running Example

```python
import gym_xarm
env = gym.make('XarmReach-v0') 
env.reset()
for _ in range(env._max_episode_timesteps):
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample())
env.close()
```

### Test Environment

In the test environment, the robot will take random actions.

```python
python test.py
```

## Demo

| XarmReach-v0                                              | XarmPickAndPlace-v0                                          | XarmPDPickAndPlace-v0                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![Large GIF (320x320)](https://tva1.sinaimg.cn/large/008i3skNgy1gsxjpl1q49g308w08wnpd.gif) | ![Large GIF (320x320)](https://tva1.sinaimg.cn/large/008i3skNgy1gsxjlnnjudg308w08wu0x.gif) | ![Large GIF (320x320)](https://tva1.sinaimg.cn/large/008i3skNgy1gsxjxkzv0tg308w08wqv5.gif) |
| XarmPDStackTower-v0                                          | XarmPDRearrange-v0                                           | XarmPDPushWithDoor-v0                                        |
| ![Large GIF (700x476)](https://tva1.sinaimg.cn/large/008i3skNly1gtcozatwg5g60jg0d84qq02.gif) | ![Large GIF (700x476)](https://tva1.sinaimg.cn/large/008i3skNly1gtcp15o238g60jg0d8e8102.gif) | ![Large GIF (700x476)](https://tva1.sinaimg.cn/large/008i3skNly1gtcsbfz94dg60jg0d8hdt02.gif) |
| XarmPDOpenBoxAndPlace-v0                                     |                                                              |                                                              |
| ![Large GIF (700x476)](https://tva1.sinaimg.cn/large/008i3skNly1gtcw0mkcerg60jg0d8hdt02.gif) |                                                              |                                                              |

> ⚠️**Note**:
>
> * `XarmPickAndPlace-v0` uses Xarm gripper, which can not be constrained in Pybullet. This will result in severe slippage or distortion in gripper shape. Both `p.createConstraint()` and `p.setJointMotorControl2()` has been tried, they are helpless in this situation even if we set a extremly large force or friction coefficient. 
> * So I recommend to use Panda gripper (the register ID is `XarmPDPickAndPlace-v0`) which needs less constrains and has a better performance in fetch tasks. 


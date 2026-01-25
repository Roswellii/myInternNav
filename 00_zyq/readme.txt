

真实测试：

启动habitat:
    conda activate habitat
    会启动实时、轨迹进展、goal的pixel
    python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_cfg.py

    结果保存在：
        ./logs/habitat/test_dual_system_0114_log/
        配置文件：
        /home/zhangwenqi/workspace/myInternNav/scripts/eval/configs/habitat_dual_system_cfg.py
    1个数据：
        /home/zhangwenqi/workspace/myInternNav/scripts/eval/configs/habitat_dual_system_single_episode_cfg.py

        
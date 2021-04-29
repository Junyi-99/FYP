import subprocess

import tvm
import os
import tvm.relay.testing
import numpy as np
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner, PSOTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from tvm import autotvm, relay
from commonf import get_network

#os.environ["TVM_BIND_THREADS"] = str(int(1))
#os.environ["TVM_NUM_THREADS"] = str(int(2))
os.environ["PATH"] = os.environ["PATH"] + ":/usr/local/cuda/bin/"

for i in range(0, 32):
    if not tvm.gpu(i).exist:
        break
    gpu = tvm.gpu(i)
    print("GPU({}): {}, Version: {}".format(i, gpu.device_name, gpu.compute_version))


def tune_kernels(
        tasks, measure_option, tuner=None, n_trial=None, early_stopping=None, log_filename=None,
        use_transter_learning=True
):
    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(task, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(task)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(task)
        elif tuner == "pso":
            tuner_obj = PSOTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        # 没错，解空间有多少，那就遍历多少（起码对于 gridsearch 是这样）
        # 其他的 Tuner 也适用于这个规则
        if n_trial is None:
            n_trial = len(task.config_space)
        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=early_stopping,  # 是否在 >= early_stopping 次搜索 时停止
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename),
            ],
        )


# Use graph tuner to achieve graph level optimal schedules
# Set use_DP=False if it takes too long to finish.
def tune_graph(graph, dshape, opt, use_DP=True):
    target_op = [
        relay.op.get("nn.conv2d"),
    ]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {opt['input_name']: dshape}, opt['log_filename'], target_op, opt['target'])
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt['graph_output_scheme_filename'])


# 该函数内无循环结构，
def perform(opt):
    print("Extract tasks ...")
    # data_shape 也被称作 input_shape
    mod, params, data_shape, out_shape = get_network(opt['network'], opt['batch_size'], dtype=opt['data_type'])
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=opt['target'], params=params, ops=(relay.op.get("nn.conv2d"),)
    )

    # 执行 Tuning 任务
    print("Tuning kernels ...")
    tune_kernels(tasks, opt['measure_option'], opt['tuner'], opt['n_trial'], opt['early_stopping'], opt['log_filename'])

    # print("Tuning graph ...")
    # tune_graph(mod['main'], data_shape, opt)

    # 根据历史最优结果编译 kernel
    # compile kernels with graph-level best records
    # history_file = opt['graph_output_scheme_filename']
    """
    history_file = opt['log_filename']
    with autotvm.apply_history_best(history_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=opt['target'], params=params)

        # load parameters
        ctx = tvm.context(str(opt['target']), 0)
        data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(opt['data_type']))
        module = tvm.runtime.GraphModule(lib["default"](ctx))
        module.set_input(opt['input_name'], data_tvm)

        # evaluate
        print("Evaluate inference time cost...")

        # repeat: 函数一共会被调用 (1 + number x repeat) 次, 第一次调用是 warmup 且结果会被舍弃，最终结果为「总调用次数-1次」结果的平均值
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=2)

        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )
    """

if __name__ == '__main__':
    for batch in [1, 16]:
        for name in [
            #'resnet-18',
            #'resnet-50', 'vgg-19', 
		'inception_v3', 'squeezenet_v1.1']:
            print("get", name, batch)
            get_network(name, batch)
            tune_opt = {
                'network': name,
                'log_filename': name + '.pso.unitune.log',
                'graph_output_scheme_filename': name + 'pso.graph.log',
                'tuner': 'pso',

                # tvm.target.cuda() or "llvm"
                # 'target': "llvm",
                'target': tvm.target.cuda(),
                'batch_size': batch,
                'data_type': 'float32',
                # Set the input name of the graph
                # For ONNX models, it is typically "0".
                'input_name': 'data',
                'n_trial': 2000,
                'early_stopping': 650,  # or an integer
                "measure_option": autotvm.measure_option(

                    builder=autotvm.LocalBuilder(timeout=10),
                    runner=autotvm.LocalRunner(
                        number=2,  # 运行多少次
                        repeat=3,  # 测量多少次。 最终运算次数为 （1 + number x repeat）
                        timeout=4,
                        min_repeat_ms=150,  # 一次运行最少持续时间（如果一个任务跑的比较快那么会多跑几次来 meet the min repeat ms）
                    ),
#                     runner=autotvm.RPCRunner(
#                         'v100',
#                         host="0.0.0.0",
#                         port=9190,
#                         number=2,
#                         repeat=3,
#                         timeout=4,
#                         min_repeat_ms=150,
#                     ),
                ),
            }
            perform(tune_opt)



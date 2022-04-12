# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import, division, print_function

import time
import paddle
from ppcls.engine.train.utils import update_loss, update_metric, log_info
from ppcls.utils import profiler


def train_epoch(engine, epoch_id, print_batch_step):
    tic = time.time()
    v_current = [int(i) for i in paddle.__version__.split(".")]
    # 修改1：训练迭代开始前，创建Profiler，设置timer_only=True
    # prof = paddle.profiler.Profiler(timer_only=True)
    # prof.start()
    for iter_id, batch in enumerate(engine.train_dataloader):
        if iter_id >= engine.max_iter:
            break
        profiler.add_profiler_step(engine.config["profiler_options"])
        if iter_id == 5:
            for key in engine.time_info:
                engine.time_info[key].reset()
        engine.time_info["reader_cost"].update(time.time() - tic)
        if engine.use_dali:
            batch = [
                paddle.to_tensor(batch[0]['data']),
                paddle.to_tensor(batch[0]['label'])
            ]
        batch_size = batch[0].shape[0]
        if not engine.config["Global"].get("use_multilabel", False):
            batch[1] = batch[1].reshape([batch_size, -1])
        engine.global_step += 1

        # image input
        if engine.amp:
            amp_level = engine.config['AMP'].get("level", "O1").upper()
            with paddle.amp.auto_cast(
                    custom_black_list={
                        "flatten_contiguous_range", "greater_than"
                    },
                    level=amp_level):
                out = forward(engine, batch)
                loss_dict = engine.train_loss_func(out, batch[1])
        else:
            out = forward(engine, batch)
            loss_dict = engine.train_loss_func(out, batch[1])

        # step opt and lr
        if engine.amp:
            scaled = engine.scaler.scale(loss_dict["loss"])
            scaled.backward()
            engine.scaler.minimize(engine.optimizer, scaled)
        else:
            loss_dict["loss"].backward()
            engine.optimizer.step()
        engine.optimizer.clear_grad()
        engine.lr_sch.step()

        # below code just for logging
        # update metric_for_logger
        update_metric(engine, out, batch, batch_size)
        # update_loss_for_logger
        update_loss(engine, loss_dict, batch_size)
        # 修改2：参考原始模型记录batch_cost的地方，调用step接口，设置num_samples为当前step的BatchSize，将记录本次迭代的开销
        # prof.step(num_samples=batch_size)
        engine.time_info["batch_cost"].update(time.time() - tic)

        if iter_id > 0 and iter_id % 65 == 0:
            engine.eval()

        if iter_id % print_batch_step == 0:
            log_info(engine, batch_size, epoch_id, iter_id)
            # step_info = prof.step_info(unit='samples')
            # print("[Train] Iter {}: {}".format(iter_id, step_info))
            # 修改3：打印使用工具计时的结果，注意unit参数的设置，比如原始log中是images/s，这里unit就是images
            # print("================== timer =================:" + prof.step_info(unit='samples'))
        tic = time.time()
    # # 修改4：在循环退出后，停止计时
    # prof.stop()


def forward(engine, batch):
    if not engine.is_rec:
        return engine.model(batch[0])
    else:
        return engine.model(batch[0], batch[1])

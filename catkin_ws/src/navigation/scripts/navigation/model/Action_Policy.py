import onnxruntime
import torch
import numpy as np

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class ONNXModel:
    def __init__(self, encoder_onnx_path,distribute_onnx_path):

        self.encoder_onnx_session = onnxruntime.InferenceSession(encoder_onnx_path)
        self.distribute_onnx_session = onnxruntime.InferenceSession(distribute_onnx_path)


    def compute(self,rgb, pointgoal, rnn_hidden_states_1, prev_actions, masks):
        rgb=to_numpy(rgb).astype(np.uint8)
        pointgoal=to_numpy(pointgoal).astype(np.float32) 
        rnn_hidden_states_1=to_numpy(rnn_hidden_states_1).astype(np.float32) 
        prev_actions=to_numpy(prev_actions).astype(np.int64)
        masks=to_numpy(masks).astype(bool)        
        # print(rgb.shape,masks,pointgoal)

        input_feed={'rgb':rgb, 'pointgoal':pointgoal, 'rnn_hidden_states.1':rnn_hidden_states_1, 'prev_actions':prev_actions, 'masks':masks}
        output_name=['features','rnn_hidden_states']
        features, rnn_hidden_states = self.encoder_onnx_session.run(output_name, input_feed=input_feed)

        input_feed={'features':features}
        output_name=['action']
        action = self.distribute_onnx_session.run(output_name, input_feed=input_feed)

        return torch.from_numpy(action[0]),torch.from_numpy(rnn_hidden_states)
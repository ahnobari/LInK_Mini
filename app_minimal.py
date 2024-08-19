import os
import uuid
import argparse
import requests
import json

argparser = argparse.ArgumentParser()
argparser.add_argument("--port", type=int, default=1238, help="Port number for the local server")
argparser.add_argument('--server', type=str, default='http://127.0.0.1:5000', help='The server to use. Default is http://127.0.0.1:5000')
argparser.add_argument("--cuda_device", type=str, default='0', help="Cuda devices to use. Default is 0")
argparser.add_argument("--static_folder", type=str, default='static', help="Folder to store static files")
# argparser.add_argument('--checkpoint_folder', type=str, default='./Checkpoints/', help='The folder to store the checkpoint')
# argparser.add_argument('--checkpoint_name', type=str, default='checkpoint.LInK', help='The name of the checkpoint file')
# argparser.add_argument('--data_folder', type=str, default='./Data/', help='The folder to store the data')
# argparser.add_argument('--embedding_folder', type=str, default='./Embeddings/', help='The folder to store the embeddings')
args = argparser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

import gradio as gr
from LInK.demo import draw_html, draw_script, css
from LInK.Solver import solve_rev_vectorized_batch_CPU
from LInK.CAD import get_layers, create_3d_html
from LInK.Optim import *
from pathlib import Path

import numpy as np
import pickle
import torch
import requests


# turn off gradient computation
torch.set_grad_enabled(False)

# check if the static folder exists
if not Path(args.static_folder).exists():
    os.mkdir(args.static_folder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load the checkpoint
# if not os.path.exists(args.checkpoint_folder) or not os.path.exists(os.path.join(args.checkpoint_folder, args.checkpoint_name)):
#     raise ValueError('The checkpoint file does not exist please run Download.py to download the checkpoints or provide the correct path.')

# # load the model
# if device == 'cpu':
#     with open(os.path.join(args.checkpoint_folder, args.checkpoint_name), 'rb') as f:
#         Trainer = pickle.load(f)
# else:
#       with open(os.path.join(args.checkpoint_folder, args.checkpoint_name), 'rb') as f:
#         Trainer = pickle.load(f)
        
# Trainer.model_base = Trainer.model_base.to('cpu')
# Trainer.model_mechanism = Trainer.model_mechanism.to('cpu')


# load data
# if not os.path.exists(args.data_folder) or not os.path.exists(os.path.join(args.data_folder, 'target_curves.npy')) or not os.path.exists(os.path.join(args.data_folder, 'connectivity.npy')) or not os.path.exists(os.path.join(args.data_folder, 'x0.npy')) or not os.path.exists(os.path.join(args.data_folder, 'node_types.npy')):
#     raise ValueError('All or some of the data does not exist please run Download.py to download the data or provide the correct path.')

# if not os.path.exists(args.embedding_folder) or not os.path.exists(os.path.join(args.embedding_folder, 'embeddings.npy')):
#     raise ValueError('The embedding file does not exist please run Download.py to download the embedding file or run Precompute.py to recompute them or provide the correct path.')

# emb  = np.load(os.path.join(args.embedding_folder, 'embeddings.npy'))[0:2000000]
# emb = torch.tensor(emb).float().to(device)
# As = np.load(os.path.join(args.data_folder, 'connectivity.npy'))[0:2000000]
# x0s = np.load(os.path.join(args.data_folder, 'x0.npy'))[0:2000000]
# node_types = np.load(os.path.join(args.data_folder, 'node_types.npy'))[0:2000000]
# curves = np.load(os.path.join(args.data_folder, 'target_curves.npy'))[0:2000000]
# sizes = (As.sum(-1)>0).sum(-1)

torch.cuda.empty_cache()

# def create_synthesizer(n_freq, maximum_joint_count, time_steps, top_n, init_optim_iters, top_n_level2, BFGS_max_iter):
#     # mask = (sizes<=maximum_joint_count)
#     synthesizer = PathSynthesis(Trainer, curves, As, x0s, node_types, emb, BFGS_max_iter=BFGS_max_iter, n_freq=n_freq, optim_timesteps=time_steps, top_n=top_n, init_optim_iters=init_optim_iters, top_n_level2=top_n_level2)
#     return synthesizer

def BFGS_Minimal(payload, progress=None, device=None, curve_size=200, smoothing=True, n_freq=5, top_n=300, max_size=20,  optim_timesteps=2000, init_optim_iters = 10, top_n_level2 = 30, CD_weight = 1.0, OD_weight = 0.25, BFGS_max_iter = 100, n_repos=1, BFGS_lineserach_max_iter=10, BFGS_line_search_mult = 0.5):
    idxs, tid, target_curve_copy, target_curve_copy_, target_curve_, target_curve, og_scale, partial, size, As, x0s, node_types = payload
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    As = torch.tensor(As).float().to(device)
    x0s = torch.tensor(x0s).float().to(device)
    node_types = torch.tensor(node_types).float().to(device)
    
    obj = make_batch_optim_obj(target_curve_copy, As, x0s, node_types,timesteps=optim_timesteps,OD_weight=OD_weight, CD_weight=CD_weight)

    prog = None
    
    x,f = Batch_BFGS(x0s, obj, max_iter=init_optim_iters, line_search_max_iter=BFGS_lineserach_max_iter, tau=BFGS_line_search_mult, progress=lambda x: demo_progress_updater(x,progress,desc='Stage 1: '),threshhold=0.001)
    
    # top n level 2
    top_n_2 = f.argsort()[:top_n_level2]
    As = As[top_n_2]
    x0s = x[top_n_2]
    node_types = node_types[top_n_2]
    
    # if partial:
    #     obj = make_batch_optim_obj_partial(target_curve_copy, target_curve_copy_, As, x0s, node_types,timesteps=self.optim_timesteps,OD_weight=0.25)
    # else:
    obj = make_batch_optim_obj(target_curve_copy, As, x0s, node_types,timesteps=optim_timesteps,OD_weight=OD_weight, CD_weight=CD_weight)
    prog2 = None
    
    for i in range(n_repos):
        x,f = Batch_BFGS(x0s, obj, max_iter=BFGS_max_iter//(n_repos+1), line_search_max_iter=BFGS_lineserach_max_iter, tau=BFGS_line_search_mult, threshhold=0.03, progress=lambda x: demo_progress_updater([x[0]/(n_repos+1) + i/(n_repos+1),x[1]],progress,desc='Stage 2: '))
        x0s = x
    
    x,f = Batch_BFGS(x0s, obj, max_iter=BFGS_max_iter - n_repos* BFGS_max_iter//(n_repos+1), line_search_max_iter=BFGS_lineserach_max_iter, tau=BFGS_line_search_mult, threshhold=0.03, progress=lambda x: demo_progress_updater([x[0]/(n_repos+1) + n_repos/(n_repos+1),x[1]],progress,desc='Stage 2: '))
    
    best_idx = f.argmin()

    end_time = time.time()
    
    if partial:
        target_uni = uniformize(target_curve_copy.unsqueeze(0),optim_timesteps)[0]
        
        tr,sc,an = find_transforms(uniformize(target_curve_.unsqueeze(0),optim_timesteps),target_uni, batch_ordered_distance)
        transformed_curve = apply_transforms(target_uni.unsqueeze(0),tr,sc,-an)[0]
        end_point = target_curve_[size-1]
        matched_point_idx = torch.argmin(torch.linalg.norm(transformed_curve-end_point,dim=-1))
        
        sol = solve_rev_vectorized_batch(As[best_idx:best_idx+1],x[best_idx:best_idx+1],node_types[best_idx:best_idx+1],torch.linspace(0,torch.pi*2,optim_timesteps).to(device))
        tid = (As[best_idx:best_idx+1].sum(-1)>0).sum(-1)-1
        best_matches = sol[np.arange(sol.shape[0]),tid]
        original_match = best_matches.clone()
        best_matches = uniformize(best_matches,optim_timesteps)
        
        tr,sc,an = find_transforms(best_matches,target_uni, batch_ordered_distance)
        tiled_curves = uniformize(target_uni[:matched_point_idx].unsqueeze(0),optim_timesteps)
        tiled_curves = apply_transforms(tiled_curves,tr,sc,-an)
        transformed_curve = tiled_curves[0].detach().cpu().numpy()
        
        best_matches = get_partial_matches(best_matches,tiled_curves[0],batch_ordered_distance)
        
        CD = batch_chamfer_distance(best_matches/sc.unsqueeze(-1).unsqueeze(-1),tiled_curves/sc.unsqueeze(-1).unsqueeze(-1))
        OD = ordered_objective_batch(best_matches/sc.unsqueeze(-1).unsqueeze(-1),tiled_curves/sc.unsqueeze(-1).unsqueeze(-1))
        
        st_id, en_id = get_partial_index(original_match,tiled_curves[0],batch_ordered_distance)
        
        st_theta = torch.linspace(0,2*np.pi,optim_timesteps).to(device)[st_id].squeeze().cpu().numpy()
        en_theta = torch.linspace(0,2*np.pi,optim_timesteps).to(device)[en_id].squeeze().cpu().numpy()
        
        st_theta[st_theta>en_theta] = st_theta[st_theta>en_theta] - 2*np.pi
        
    else:
        sol = solve_rev_vectorized_batch(As[best_idx:best_idx+1],x[best_idx:best_idx+1],node_types[best_idx:best_idx+1],torch.linspace(0,torch.pi*2,optim_timesteps).to(device))
        tid = (As[best_idx:best_idx+1].sum(-1)>0).sum(-1)-1
        best_matches = sol[np.arange(sol.shape[0]),tid]
        best_matches = uniformize(best_matches,optim_timesteps)
        target_uni = uniformize(target_curve_copy.unsqueeze(0),optim_timesteps)[0]
        
        tr,sc,an = find_transforms(best_matches,target_uni, batch_ordered_distance)
        tiled_curves = uniformize(target_curve_copy.unsqueeze(0),optim_timesteps)
        tiled_curves = apply_transforms(tiled_curves,tr,sc,-an)
        transformed_curve = tiled_curves[0].detach().cpu().numpy()
        
        CD = batch_chamfer_distance(best_matches/sc.unsqueeze(-1).unsqueeze(-1),tiled_curves/sc.unsqueeze(-1).unsqueeze(-1))
        OD = ordered_objective_batch(best_matches/sc.unsqueeze(-1).unsqueeze(-1),tiled_curves/sc.unsqueeze(-1).unsqueeze(-1))
        
        st_theta = 0.
        en_theta = np.pi*2
    
    n_joints = (As[best_idx].sum(-1)>0).sum().cpu().numpy()
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    ax = draw_mechanism(As[best_idx].cpu().numpy()[:n_joints,:][:,:n_joints],x[best_idx].cpu().numpy()[0:n_joints],np.where(node_types[best_idx].cpu().numpy()[0:n_joints])[0],[0,1],highlight=tid[0].item(),solve=True, thetas=np.linspace(st_theta,en_theta,optim_timesteps),ax=ax)
    ax.plot(transformed_curve[:,0], transformed_curve[:,1], color="indigo", alpha=0.7, linewidth=2)
    
    A = As[best_idx].cpu().numpy()
    x = x[best_idx].cpu().numpy()
    node_types = node_types[best_idx].cpu().numpy()
    
    n_joints = (A.sum(-1)>0).sum()
    
    A = A[:n_joints,:][:,:n_joints]
    x = x[:n_joints]
    node_types = node_types[:n_joints]
    
    transformation = [tr.cpu().numpy(),sc.cpu().numpy(),an.cpu().numpy()]
    start_theta = st_theta
    end_theta = en_theta
    performance = [CD.item()*og_scale.cpu().numpy(),OD.item()*(og_scale.cpu().numpy()**2),og_scale.cpu().numpy()]
    torch.cuda.empty_cache()
    return fig, [[A,x,node_types, start_theta, end_theta, transformation], performance, transformed_curve], gr.update(value = {"Progress":1.0})

def make_cad(synth_out, partial, progress=gr.Progress(track_tqdm=True)):
    
    progress(0, desc="Generating 3D Model ...")
    
    f_name = str(uuid.uuid4())
    
    A_M, x0_M, node_types_M, start_theta_M, end_theta_M, tr_M = synth_out[0]
    
    sol_m = solve_rev_vectorized_batch_CPU(A_M[np.newaxis], x0_M[np.newaxis],node_types_M[np.newaxis],np.linspace(start_theta_M, end_theta_M, 200))[0]
    z,status = get_layers(A_M, x0_M, node_types_M,sol_m)
    
    if partial:
        sol_m = np.concatenate([sol_m, sol_m[:,::-1,:]], axis=1)
    
    create_3d_html(A_M, x0_M, node_types_M, z, sol_m, template_path = f'./{args.static_folder}/animation.html', save_path=f'./{args.static_folder}/{f_name}.html')
    
    return gr.HTML(f'<iframe width="100%" height="800px" src="file={args.static_folder}/{f_name}.html"></iframe>',label="3D Plot",elem_classes="plot3d")


def setup(n_freq, partial, test_curve,top_n, maximum_joint_count):
    package = {}
    print('setup ...')

    package['target_curve'] = test_curve.tolist()
    package['partial'] = partial
    package['curve_size'] = 200
    package['smoothing'] = True
    package['n_freq'] = n_freq
    package['top_n'] = top_n
    package['max_size'] = maximum_joint_count

    return package

def step_1(package):
    
    url = args.server
    package = json.dumps(package)
    response = requests.post(url, data=package)

    payload = pickle.loads(response.content)

    print('payload obtained ...')

    return payload[:-3], payload[-3], payload[-2], payload[-1]

def step_2(payload, time_steps, top_n, init_optim_iters, top_n_level2, BFGS_max_iter, n_freq, maximum_joint_count, progress=gr.Progress()):
    return BFGS_Minimal(payload, progress=progress, device='cuda', smoothing=True, n_freq=n_freq, top_n=top_n, max_size=maximum_joint_count,  optim_timesteps=time_steps, init_optim_iters = init_optim_iters, top_n_level2 = top_n_level2, BFGS_max_iter = BFGS_max_iter)
    

gr.set_static_paths(paths=[Path(f'./{args.static_folder}')])


with gr.Blocks(css=css, js=draw_script) as block:
    
    syth  = gr.State()
    state = gr.State()
    dictS = gr.State(False)
    
    with gr.Row():
        with gr.Column(min_width=350,scale=2):
            canvas = gr.HTML(draw_html)
            clr_btn = gr.Button("Clear",elem_classes="clr_btn")
            
            btn_submit = gr.Button("Perform Path Synthesis",variant='primary',elem_classes="clr_btn")
            
            # checkbox
            partial = gr.Checkbox(label="Partial Curve", value=False, elem_id="partial")

        with gr.Column(min_width=250,scale=1,visible=True):
            gr.HTML("<h2>Algorithm Parameters</h2>")
            
            n_freq = gr.Slider(minimum = 3 , maximum = 50, value=7, step=1, label="Number of Frequenceies For smoothing", interactive=True)
            maximum_joint_count = gr.Slider(minimum = 6 , maximum = 20, value=14, step=1, label="Maximum Joint Count", interactive=True)
            time_steps = gr.Slider(minimum = 200 , maximum = 5000, value=2000, step=1, label="Number of Simulation Time Steps", interactive=True)
            top_n = gr.Slider(minimum = 50 , maximum = 1000, value=300, step=1, label="Top N Candidates To Start With", interactive=True)
            init_optim_iters = gr.Slider(minimum = 10 , maximum = 50, value=20, step=1, label="Initial Optimization Iterations On All Candidates", interactive=True)
            top_n_level2 = gr.Slider(minimum = 10 , maximum = 100, value=30, step=1, label="Top N Candidates For Final Optimization", interactive=True)
            BFGS_max_iter = gr.Slider(minimum = 50 , maximum = 500, value=200, step=1, label="Iterations For Final Optimization", interactive=True)
        
    with gr.Row():
        with gr.Row():
            with gr.Column(min_width=250,scale=1,visible=True):
                gr.HTML('<h2>Algorithm Outputs</h2>')
                progl = gr.Label({"Progress": 0}, elem_classes="prog",num_top_classes=1)
                
    with gr.Row():
        with gr.Column(min_width=250,visible=True):
            og_plt = gr.Plot(label="Original Input",elem_classes="plotpad")
        with gr.Column(min_width=250,visible=True):    
            smooth_plt = gr.Plot(label="Smoothed Drawing",elem_classes="plotpad")
                
    with gr.Row():
        candidate_plt = gr.Plot(label="Initial Candidates",elem_classes="plotpad")
    
    with gr.Row():
        mechanism_plot = gr.Plot(label="Solution",elem_classes="plotpad")
        

    with gr.Row():
        plot_3d = gr.HTML('<iframe width="100%" height="800px" src="file=static/filler.html"></iframe>',label="3D Plot",elem_classes="plot3d")
    
    event1 = btn_submit.click(lambda: [None]*4 + [gr.update(interactive=False)]*8, outputs=[candidate_plt,mechanism_plot,og_plt,smooth_plt,btn_submit, n_freq, maximum_joint_count, time_steps, top_n, init_optim_iters, top_n_level2, BFGS_max_iter])
    # event2 = event1.then(create_synthesizer, inputs=[n_freq, maximum_joint_count, time_steps, top_n, init_optim_iters, top_n_level2, BFGS_max_iter], outputs=[syth], concurrency_limit=1)
    event2 = event1.then(lambda a,b,c,e,f: setup(a,b,np.array([eval(i) for i in c.split(',')]).reshape([-1,2]) * [[1,-1]],e,f), inputs=[n_freq, partial,canvas,top_n, maximum_joint_count], outputs=[syth], js="(a,b,c,e,f) => [a,b,path.toString(),e,f]")
    event3 = event2.then(step_1, inputs=[syth], outputs=[state,candidate_plt,og_plt,smooth_plt], concurrency_limit=1)
    event4 = event3.then(step_2, inputs=[state, time_steps, top_n, init_optim_iters, top_n_level2, BFGS_max_iter, n_freq, maximum_joint_count], outputs=[mechanism_plot,state,progl], concurrency_limit=1)
    event5 = event4.then(make_cad, inputs=[state,partial], outputs=[plot_3d], concurrency_limit=1)
    event6 = event5.then(lambda: [gr.update(interactive=True)]*8, outputs=[btn_submit, n_freq, maximum_joint_count, time_steps, top_n, init_optim_iters, top_n_level2, BFGS_max_iter], concurrency_limit=1)
    
    # event3 = event2.then(lambda s,x,p: s.demo_sythesize_step_1(np.array([eval(i) for i in x.split(',')]).reshape([-1,2]) * [[1,-1]],partial=p), inputs=[syth,canvas,partial],js="(s,x,p) => [s,path.toString(),p]",outputs=[state,og_plt,smooth_plt], concurrency_limit=1)
    # event4 = event3.then(lambda sy,s,mj: sy.demo_sythesize_step_2(s,max_size=mj), inputs=[syth,state,maximum_joint_count], outputs=[state,candidate_plt], concurrency_limit=1)
    # event5 = event4.then(lambda sy,s: sy.demo_sythesize_step_3(s,progress=gr.Progress()), inputs=[syth,state], outputs=[mechanism_plot,state,progl], concurrency_limit=1)
    # event6 = event5.then(make_cad, inputs=[state,partial], outputs=[plot_3d], concurrency_limit=1)
    # event8 = event6.then(lambda: [gr.update(interactive=True)]*8, outputs=[btn_submit, n_freq, maximum_joint_count, time_steps, top_n, init_optim_iters, top_n_level2, BFGS_max_iter], concurrency_limit=1)
    block.load()
    
    clr_btn.click(lambda x: x, js='document.getElementById("sketch").innerHTML = ""')
    
block.launch(server_port=args.port,share=True,max_threads=200,inline=False)
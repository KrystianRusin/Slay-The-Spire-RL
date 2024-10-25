from multiprocessing import Process, Queue
from sb3_contrib.ppo_mask import MaskablePPO
from slay_the_spire_env import SlayTheSpireEnv
from environment.run_env import run_environment
from model.model_utils import update_model
import torch as th

def main():
    num_envs = 4
    base_port = 9999
    experience_queue = Queue()
    n_steps = 2048
    processes = []

    for env_id in range(num_envs):
        port = base_port + env_id
        p = Process(target=run_environment, args=(env_id, port, experience_queue))
        p.start()
        processes.append(p)

    def linear_clip_range(progress_remaining):
       
        return 0.3 * progress_remaining

    model = MaskablePPO(
        "MultiInputPolicy",
        SlayTheSpireEnv({}),
        ent_coef=0.03,
        gamma=0.97,
        learning_rate=0.0003,
        clip_range=linear_clip_range,
        verbose=1,
        device=th.device("cuda" if th.cuda.is_available() else "cpu")
    )
    
    total_steps = 100000
    current_step = 0
    
    while current_step < total_steps:
        experiences = []
        for _ in range(num_envs):
            experiences.append(experience_queue.get())

        for exp in experiences:
            update_model(model, exp, current_step, total_steps)

        model.save("maskable_ppo_slay_the_spire")
        current_step += n_steps * num_envs

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()

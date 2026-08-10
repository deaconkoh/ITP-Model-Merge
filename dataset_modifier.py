import random
import os


for file in os.listdir("data\\SD2\\10x5+mix"):
    with open(f"data\\SD2\\10x5+mix\\{file}", "r") as input_file:
        file_str = ""
        jobs = input_file.readlines()
        for job in jobs[1:]:
            job = job[:-1].split(" ")
            job_str = f"{job[0]} "
            i = 1
            while i < len(job):
                x = int(job[i])
                priority = random.randint(1, 100)
                op_str = f"{x} {priority} "
                for j in range(x):
                    carbon = random.randint(1, 100)
                    op_str += f"{job[1+i+j*2]} {job[2+i+j*2]} {carbon} "
                job_str += op_str
                i += 1 + x * 2
            file_str += job_str + "\n"
        with open(f"data\\SD2\\10x5+carbon+priority\\{file}", "w") as output_file:
            output_file.write(jobs[0])
            output_file.write(file_str)

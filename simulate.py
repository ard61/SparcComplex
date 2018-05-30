def run_simulation(sparc_params_json, num_trials):
    import sys
    sys.path.append('/home/zelda/ard61/project/code')

    import numpy as np
    import sparc as sp

    p = sp.SparcParams.from_json(sparc_params_json)

    ## Setup
    if p.D == 1:
        (Ax, Ay) = sp.sparc_transforms(p.n, p.M, p.L, sp.gen_ordering(p.n, p.M, p.L))
        if p.logK == 0:
            sparc = sp.Sparc(p, Ax, Ay)
        else:
            sparc = sp.SparcModulated(p, Ax, Ay)
    else:
        (Ax, Ay) = sp.complex_sparc_transforms(p.n, p.M, p.L, sp.gen_ordering(p.n, p.M, p.L))
        sparc = sp.SparcQpsk(p, Ax, Ay)

    CERs = np.zeros(num_trials)
    SERs = np.zeros(num_trials)
    BERs = np.zeros(num_trials)

    for i in range(num_trials):
        CERs[i], SERs[i], BERs[i] = sp.error_rate(sparc)

    return sp.DataPoint(p, num_trials, CERs, SERs, BERs)

def simulate(sparc_params, num_trials):
    import sys
    sys.path.append('/home/zelda/ard61/project/code')

    import sheepdog
    import sparc as sp
    
    #conf = {"host": "yoshi", "shell": "/usr/bin/python3", "ge_opts": ['-q yoshi-low.q@yoshi.eng.cam.ac.uk']}
    conf = {"host": "yoshi", "shell": "/usr/bin/python3"}

    sparc_params_json = sparc_params.to_json()
    
    num_machines = 40
    quotient = num_trials // num_machines
    remainder = num_trials - num_machines * quotient
    trial_alloc = [quotient + 1] * remainder + [quotient] * (num_machines - remainder)
    args = [[sparc_params_json, x] for x in trial_alloc]

    results = sheepdog.map(run_simulation, args, conf)
    data_point = sp.DataPoint.combine([result if result is not None 
                                              else sp.DataPoint.none(sparc_params)
                                       for result in results])
    
    return data_point

executable = /afs/desy.de/user/a/aulich/mu3e_trigger/CONDOR/submitCondor.sh
universe = vanilla
arguments = "python3 scripts/train_contrastive_triplet.py"
error = /afs/desy.de/user/a/aulich/mu3e_trigger/CONDOR/logs/$(ClusterId).$(ProcId).err
log = /afs/desy.de/user/a/aulich/mu3e_trigger/CONDOR/logs/$(ClusterId).$(ProcId).log
output = /afs/desy.de/user/a/aulich/mu3e_trigger/CONDOR/logs/$(ClusterId).$(ProcId).out
RequestCPUs = 8
RequestGPUs = 1
RequestMemory = 40000
+RequestRuntime = 100000
+MaxRuntime = 100000
transfer_executable = False
should_transfer_files = False
checkpoint = True

queue  1
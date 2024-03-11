import stick

run_dir = stick.init_extra()
print("Logging experiment results to", run_dir)
stick.log("test_table", {"x": 0.0})

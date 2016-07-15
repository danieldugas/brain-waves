def load_raw_waves(folder="/home/daniel/Downloads/Raw-Waves/", filename="001_Session1_FilterTrigCh_RawCh.mat"):
  import scipy.io
  # folder = "/home/daniel/Downloads/Waves 37.5 up down/37.5uV up down/"
  # filename = "001_Session1_waves.mat"
  mat = scipy.io.loadmat(folder+filename)
  wave = mat.get('wave')[0]
  raw =mat.get('data_raw')[0]
  data = mat.get('data')[0]

  return data

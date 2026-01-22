import sys
import os
import time
import datetime
import signal
import argparse
import csv

import h5py
import numpy as np

import pymoduleconnector
from pymoduleconnector import ModuleConnector
from pymoduleconnector import DataType
from pymoduleconnector import PreferredSplitSize
from pymoduleconnector import RecordingOptions
from pymoduleconnector import DataRecorder
from pymoduleconnector import create_mc
from pymoduleconnector.ids import *


class HDF5PacketWriter:
    def __init__(self, _datatype, h5_path):
        self.h5_path = h5_path
        self._datatype = _datatype
        vlen_float64 = h5py.special_dtype(vlen=np.float64)

        # 根据要录制的数据，定义结构化 dtype
        if self._datatype == "baseband":
            self.dtype = np.dtype([
                ('timestamp', 'S32'),
                ('frame_counter', 'u4'),
                ('num_bins', 'u4'),
                ('bin_length', 'f8'),
                ('sample_frequency', 'f8'),
                ('carrier_frequency', 'f8'),
                ('range_offset', 'f8'),
                ('I_data', vlen_float64),
                ('Q_data', vlen_float64)
            ])
        elif self._datatype == "doppler":
            self.dtype = np.dtype([
                ('timestamp', 'S32'),
                ('frame_counter', 'u4'),
                ('matrix_counter', 'u4'),
                ('range_idx', 'u4'),
                ('range_bins', 'u4'),
                ('pulsedoppler_instance', 'u4'),  
                ('frequency_count', 'u4'), 
                ('fps', 'f4'), 
                ('fps_decimated', 'f4'), 
                ('frequency_start', 'f8'), 
                ('frequency_step', 'f8'), 
                ('range', 'f8'), 
                ('Doppler_power', vlen_float64),               

            ])


        # 创建 HDF5 文件和可扩展 dataset
        self.h5file = h5py.File(h5_path, 'w')
        self.dataset = self.h5file.create_dataset(
            self._datatype,
            shape=(0,), maxshape=(None,),
            dtype=self.dtype,
            compression="gzip"
        )
        # self.h5file.attrs['num_bins'] = num_bins  # 记录元数据
        self.index = 0

    def append_packet(self, pkt):
        if self._datatype == "baseband":

            row = np.array(
                # (datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'),
                (datetime.datetime.now().timestamp(),
                pkt.frame_counter,
                pkt.num_bins,
                pkt.bin_length,
                pkt.sample_frequency,
                pkt.carrier_frequency,
                pkt.range_offset,
                np.array(pkt.get_I(), dtype=np.float64),
                np.array(pkt.get_Q(), dtype=np.float64)),
                dtype=self.dtype
            )

        if self._datatype == "doppler":

            row = np.array(
                (datetime.datetime.now().timestamp(),
                pkt.frame_counter,
                pkt.matrix_counter,
                pkt.range_idx,
                pkt.range_bins,
                pkt.pulsedoppler_instance,
                pkt.frequency_count,
                pkt.fps,
                pkt.fps_decimated,
                pkt.frequency_start,
                pkt.frequency_step,
                pkt.range,
                np.array(pkt.get_data(), dtype=np.float64)),
                dtype=self.dtype
            )

        self.dataset.resize(self.index + 1, axis=0)
        self.dataset[self.index] = row
        self.index += 1

    def close(self):
        self.h5file.close()

class X4m300DataRecord():
    def __init__(self, com_port:str, detection_zone: tuple=(0.4, 5.0), sensitive: int = 9, noisemap_mode: str="default"):
        """初始化x4m300和xep, 并设置noise map.

        Args:
            com_port (string): x4m300雷达传感器的连接端口
            detection_zone (tuple): 雷达的检测范围, 其限制为(0.4000000059604645 9.399999618530273)
            sensitive (int): 设置sensitive, 其范围为0到9
            nosemap_mode: (string): nosemap的模式
        """
        mc = ModuleConnector(com_port)
        self.xep = mc.get_xep()
        self.x4m300 = mc.get_x4m300()
        self.detection_zone_min, self.detection_zone_max = detection_zone
        self.noisemap_mode = noisemap_mode
        self._stop = False
        
        
    def configue_noisemap(self, enable_adaptive_noisemap: bool=True):
        profile = XTS_ID_APP_PRESENCE_2
        self.x4m300.load_profile(profile)
        self.x4m300.set_detection_zone(self.detection_zone_min, self.detection_zone_max)
        
        self.x4m300.set_noisemap_control(0) #set the nois_map control to all 0.

        if self.noisemap_mode == "default":
            ctl = XTID_NOISEMAP_CONTROL_USE_DEFAULT | XTID_NOISEMAP_CONTROL_INIT_ON_RESET # 0b110(6) is to use the default noisemap.
        elif self.noisemap_mode == "stored":
            print("The stored noise map is used, please confirm the stored noisemap is valid. If the stored noisemap is invalid, the a noisemap will be generated and stored")
            ctl = XTID_NOISEMAP_CONTROL_ENABLE  # 0b011 (3)
            ctl &= ~XTID_NOISEMAP_CONTROL_INIT_ON_RESET   
        elif self.noisemap_mode == "initilize":
            print("Initilizing a new noise map, the stored noise map will not be overwritten.")
            ctl = XTID_NOISEMAP_CONTROL_ENABLE | XTID_NOISEMAP_CONTROL_INIT_ON_RESET #0b0111 (7)
        if enable_adaptive_noisemap:
            ctl = ctl | XTID_NOISEMAP_CONTROL_ADAPTIVE # 默认开启，因此从右向左第二位默认为1
        else:
            ctl = ctl | XTID_NOISEMAP_CONTROL_NONADAPTIVE

        self.x4m300.set_noisemap_control(ctl)

        
    def _list_noisemap_file(self):
        files = self.xep.find_all_files()
        return set(zip(files.file_type_items, files.file_identifier_items))
    
    
    def _delete_stored_noisemap(self):
        files = self._list_noisemap_file()

        if (XTFILE_TYPE_NOISEMAP_FAST, 0) in files:
            self.xep.delete_file(XTFILE_TYPE_NOISEMAP_FAST, 0)
        if (XTFILE_TYPE_NOISEMAP_SLOW, 0) in files:
            self.xep.delete_file(XTFILE_TYPE_NOISEMAP_SLOW, 0)
            
        if ((XTFILE_TYPE_NOISEMAP_FAST, 0) not in files) and ((XTFILE_TYPE_NOISEMAP_SLOW, 0) not in files):
            print("The stored nosiemap has been deleted!")
           
    def _flush_doppler_data_buffer(self):
        while self.x4m300.peek_message_pulsedoppler_byte():
            self.x4m300.read_message_pulsedoppler_byte()
        while self.x4m300.peek_message_pulsedoppler_float():
            self.x4m300.read_message_pulsedoppler_float()
        # while self.x4m300.peek_message_noisemap_byte(): # This will delete the noismap don't do this.
        #     self.x4m300.read_message_noisemap_byte()
        # while self.x4m300.peek_message_noisemap_float():
        #     self.x4m300.read_message_noisemap_float()
        
    
    def _flush_baseband_data_buffer(self):
        while self.x4m300.peek_message_baseband_ap():
            self.x4m300.read_message_baseband_ap()
        while self.x4m300.peek_message_baseband_iq():
            self.x4m300.read_message_baseband_iq()
            
    def _flush_movinglist_data_buffer(self):
        while self.x4m300.peek_message_presence_movinglist():
            self.x4m300.read_message_presence_movinglist()
            
    
    def start_sensor(self):
        if self.noisemap_mode == "default":
            print("Starting the sensor with default noisemap, please wait for ~30seconds.")
            time.sleep(5)
            status = self.x4m300.set_sensor_mode(XTID_SM_RUN, 0)
            time.sleep(30)
        else:
            print("Starting the sensor with stored noisemap or initilization, please waith for ~2min.")
            time.sleep(5)
            status = self.x4m300.set_sensor_mode(XTID_SM_RUN, 0)
            time.sleep(130)
            
        print("Sensor started")
  
        
    def stop_sensor(self):
        self.x4m300.set_sensor_mode(XTID_SM_STOP, 0)
        print(f"Sensor is stopped")
        
            
    def prepare_record_doppler(self, output: str="both", format: str="float"):
        self._flush_doppler_data_buffer()
        self.doppler_format = format
        
        if output == "both":
            ctrl = XTID_OUTPUT_CONTROL_PD_FAST_ENABLE | XTID_OUTPUT_CONTROL_PD_SLOW_ENABLE
        elif output == "fast":                        # the pkt.pulsedoppler_instance == 0 if the pkt belongs to fast doppler map.
            ctrl = XTID_OUTPUT_CONTROL_PD_FAST_ENABLE
        elif output == "slow":                        # the pkt.pulsedoppler_instance == 1 if the pkt belongs to slow doppler map.
            ctrl = XTID_OUTPUT_CONTROL_PD_SLOW_ENABLE
        else:
            print("Pulse-Doppler instance not recognized.", file=sys.stderr)
            raise SystemExit(1)
        
        if format == "float":
            self.x4m300.set_output_control(XTS_ID_PULSEDOPPLER_FLOAT, ctrl)
        elif format == "byte":
            self.x4m300.set_output_control(XTS_ID_PULSEDOPPLER_BYTE, ctrl)
            
        
    def prepare_record_baseband(self, output: str="iq"):
        self._flush_baseband_data_buffer()
        self.baseband_output = output
        ctrl = XTID_OUTPUT_CONTROL_ENABLE
        if output == "ap":
            self.x4m300.set_output_control(XTS_ID_BASEBAND_AMPLITUDE_PHASE, ctrl)
        elif output == "iq":
            self.x4m300.set_output_control(XTS_ID_BASEBAND_IQ, ctrl)
        else:
            print("Baseband instance not recognized.", file=sys.stderr)
            raise SystemExit(1)

        
    def prepare_record_movinglist(self):
        self._flush_movinglist_data_buffer()
        ctrl = XTID_OUTPUT_CONTROL_ENABLE
        self.x4m300.set_output_control(XTS_ID_PRESENCE_MOVINGLIST, ctrl)

       
    def _record_doppler(self, writer):
             pkt_doppler = (self.x4m300.read_message_pulsedoppler_float() 
                            if self.doppler_format == "float"
                            else self.x4m300.read_message_pulsedoppler_byte())
             writer.append_packet(pkt_doppler)
             
    def _record_baseband(self, writer):
            pkt_baseband = (self.x4m300.read_message_baseband_ap()
                            if self.baseband_output == "ap"
                            else self.x4m300.read_message_baseband_iq())
            writer.append_packet(pkt_baseband)
 
            
    def _record_movinglist(self, writer):
            pkt_movinglist = self.x4m300.read_message_presence_movinglist()
            writer.writerow([datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'), pkt_movinglist.frame_counter, pkt_movinglist.presence_state, pkt_movinglist.get_movement_slow_items(), 
                             pkt_movinglist.get_movement_fast_items(), pkt_movinglist.get_radar_cross_section(), pkt_movinglist.get_detection_distance_items(),
                             pkt_movinglist.get_detection_velocity_items()])
 
                
    def record_data(self, _path:str = None): #doppler_path: str = None, baseband_path: str = None, movinglist_path: str = None):
        self.start_sensor()
        writers = {}
        _time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        _update_path = os.path.join(_path, _time)
        os.makedirs(_update_path, exist_ok=True)
        doppler_path = os.path.join(_update_path, "doppler.h5")
        baseband_path = os.path.join(_update_path, "baseband.h5")
        movinglist_path = os.path.join(_update_path, "movlinglist.csv")
        def stop_all(signum, frame):
            print("\nStopping sensor and threads...", file=sys.stderr)
            self._stop = True

        signal.signal(signal.SIGINT, stop_all)
        signal.signal(signal.SIGTERM, stop_all)

        if doppler_path:
            writer = HDF5PacketWriter(_datatype="doppler", h5_path=doppler_path)
            writers["doppler"] = writer

        if baseband_path:
            writer = HDF5PacketWriter(_datatype="baseband", h5_path=baseband_path)
            writers["baseband"] = writer
                        
        if movinglist_path:
            f_movinglist = open(movinglist_path, "w+", newline="")
            writer = csv.writer(f_movinglist)
            writer.writerow([
                            "TimeStamp", "FrameCounter", "PresenceState", "MovementSlowItems", \
                            "MovementFastItems", "DetectionRadarCrossSection", "DetectionDistance", "DetectionVelocity"
            ])
            writers["movinglist"] = writer

        try:
            while not self._stop:
                did_work = False
                if doppler_path is not None and self.x4m300.peek_message_pulsedoppler_float():
                    did_work = True
                    self._record_doppler(writer=writers["doppler"])

                if baseband_path is not None and self.x4m300.peek_message_baseband_iq():
                    did_work = True
                    self._record_baseband(writer=writers["baseband"])

                if movinglist_path is not None and  self.x4m300.peek_message_presence_movinglist():
                    did_work = True
                    self._record_movinglist(writer=writers["movinglist"])
                    
                if not did_work:
                    time.sleep(0.0001)
                
        finally:
            self.stop_sensor()
            for t, writer_ in writers.items():
                if t != "movinglist":
                    writer_.close()

            f_movinglist.flush()
            f_movinglist.close()
            
            print("Sensor and all threads stopped. Exiting.")    
    
        
    
    
if __name__ == "__main__":
    radar_instance = X4m300DataRecord("COM7", noisemap_mode="default", detection_zone=(0.41, 4.0))
    radar_instance.configue_noisemap(enable_adaptive_noisemap=False)
    radar_instance.prepare_record_baseband()
    radar_instance.prepare_record_doppler()
    radar_instance.prepare_record_movinglist()

    radar_instance.record_data(_path = "E:/multi_sensor_sync_save_data_test/x4m300"
                                )
    # "F:/X4M300_radar/data/test/doppler_test_2.csv", "F:/X4M300_radar/data/test/baseband_test_2.csv"
    
    
        




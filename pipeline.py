import os
import logging

from pipeline_processor import PipelineProcessor

class PipelineRunner(object):

    def __init__(self, pipeline=None, log_level=logging.DEBUG):
        self.pipeline = pipeline or []
        self.context = {}
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(log_level)
        self.log_level = log_level
        self.set_log_level()
    

    def set_context(self, data):
        self.context = data


    def add(self, processor):
        if not isinstance(processor, PipelineProcessor):
            raise Exception('Processor should be instance of PipelineProcessor')
        processor.log.setLevel(self.log_level)
        self.pipeline.append(processor)
    

    def remove(self, name):
        for i, p in enumerate(self.pipeline):
            if p.__class__.__name__ == name:
                del self.pipeline[i]
                return True
        return False

    
    def set_log_level(self):
        for p in self.pipeline:
            p.log.setLevel(self.log_level)


    def run(self):
        for p in self.pipeline:
            self.context = p(self.context)

        self.log.debug("Frame #%d processed.", self.context['frame_number'])

        return self.context
from TTS.api import TTS


class Models_Examples:

    @staticmethod
    def xtts_from_sample_audio_and_text_to_output_audio(device, text, audio_sample_path, lang, output_path):
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        tts.tts_to_file(text=text, 
                speaker_wav=audio_sample_path, 
                language=lang, 
                file_path=output_path,
                split_sentences=True)
        
    @staticmethod
    def vctk_from_audio_to_audio_output(device, source_sample_path, target_sample_path, output_path):
        tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to(device)
        tts.voice_conversion_to_file(source_wav=source_sample_path,
                                    target_wav=target_sample_path, 
                                    file_path=output_path)     
    
    @staticmethod
    def tacotron2_GER_from_text_to_output_audio(device, text, output_path):
        tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False).to(device)
        tts.tts_to_file(text=text, 
                        file_path=output_path)
        
    @staticmethod
    def your_tts_from_sample_audio_and_text_to_output_audio(device, text, audio_sample_path, lang, output_path):
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)
        tts.tts_to_file(text=text, 
                speaker_wav=audio_sample_path, 
                language=lang, 
                file_path=output_path,
                split_sentences=True)
    @staticmethod
    def fairseq_from_sample_audio_and_text_to_output_audio(text, audio_sample_path, lang, output_path):
        # https://github.com/facebookresearch/fairseq/tree/main/examples/mms/tts/tutorial
        api = TTS(f"tts_models/{lang}/fairseq/vits")
        api.tts_with_vc_to_file(
            text,
            speaker_wav=audio_sample_path,
            file_path=output_path )     

    @staticmethod
    # Probar con tortoise
    def xtts_from_text_to_output_audio(device, text, lang, output_path):
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        tts.tts_to_file(text=text, language=lang, 
                        file_path=output_path, 
                        split_sentences=True)
import React, { useState, useRef } from 'react';
import { useForm } from 'react-hook-form';
import { FiImage, FiLoader, FiUpload, FiActivity, FiZap, FiCrosshair } from 'react-icons/fi';
import { motion } from 'framer-motion';
import { useTheme } from '../context/ThemeContext';
import DashboardLayout from '../components/layout/DashboardLayout';
import FormContainer from '../components/ui/FormContainer';
import PredictionResult from '../components/ui/PredictionResult';
import { predictDisease } from '../utils/api';

interface BreastCancerFormData {
  image: FileList;
}

const BreastCancerPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any | null>(null);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const { theme, accentColor } = useTheme();

  const { register, handleSubmit, reset } = useForm<BreastCancerFormData>();
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setPreviewImage(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const onSubmit = async (data: BreastCancerFormData) => {
    if (!data.image || data.image.length === 0) return;
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('image', data.image[0]);

      const response = await predictDisease.breast(formData);

      if (response.error) {
        setResult({
          error: true,
          prediction: 'Analysis Failed',
          confidence: 0,
          model_name: 'Unknown',
          metrics: { accuracy: 0, precision: 0, recall: 0, f1: 0 }
        });
        setLoading(false);
        return;
      }

      setResult({
        prediction: response.class_name,
        confidence: response.confidence,
        model_name: response.model_name,
        metrics: {
          accuracy: response.accuracy,
          precision: response.precision,
          recall: response.recall,
          f1: response.f1
        }
      });
    } catch (error) {
      setResult({
        error: true,
        prediction: 'Analysis Failed',
        confidence: 0,
        model_name: 'Unknown',
        metrics: { accuracy: 0, precision: 0, recall: 0, f1: 0 }
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <DashboardLayout>
      <FormContainer
        title="Breast Cancer Check"
        subtitle="Analyze for potential cells"
        icon={<FiCrosshair size={28} />}
        predictionResult={result && (
          <div className="space-y-6">
            <PredictionResult
              prediction={result.prediction}
              confidence={result.confidence}
              modelName={result.model_name}
              metrics={result.metrics}
              diseaseType="breast"
              error={result.error}
            />
            <button onClick={() => { setPreviewImage(null); setResult(null); reset(); }} className="w-full py-4 rounded-2xl font-bold border-2 transition-all hover:bg-slate-50 dark:hover:bg-white/5 active:scale-95" style={{ borderColor: accentColor, color: accentColor }}>
              Check New Image
            </button>
          </div>
        )}
      >
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-10 flex flex-col justify-center h-full pb-10">
          <div className="text-center space-y-4">
            <div className="w-16 h-16 rounded-2xl bg-slate-50 dark:bg-white/5 flex items-center justify-center mx-auto text-[#14b8a6] shadow-inner">
              <FiZap size={32} />
            </div>
            <h3 className="text-2xl font-black tracking-tight">Upload Medical Photo</h3>
            <p className="text-sm opacity-50 font-medium max-w-sm mx-auto">Upload a clear photo of the test area to check for health risks.</p>
          </div>

          <motion.div
            className="w-full max-w-md mx-auto aspect-square rounded-[2rem] border-2 border-dashed flex flex-col items-center justify-center cursor-pointer overflow-hidden relative transition-all group"
            whileHover={{ scale: 1.02 }}
            style={{
              borderColor: previewImage ? accentColor : `${accentColor}40`,
              backgroundColor: theme === 'dark' ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.4)'
            }}
            onClick={() => fileInputRef.current?.click()}
          >
            {previewImage ? (
              <img src={previewImage} alt="Preview" className="w-full h-full object-cover" />
            ) : (
              <div className="text-center p-8 space-y-4">
                <div className="w-20 h-20 rounded-3xl mx-auto flex items-center justify-center bg-white dark:bg-black/20 shadow-xl text-slate-400 group-hover:text-[#14b8a6] transition-colors">
                  <FiUpload size={32} />
                </div>
                <p className="font-bold">Select Photo</p>
                <p className="text-xs opacity-50">JPG or PNG photos supported</p>
              </div>
            )}
            <input type="file" className="hidden" accept="image/*" {...register('image', { onChange: handleImageChange })} ref={(e) => { register('image').ref(e); fileInputRef.current = e; }} />
          </motion.div>

          <button
            type="submit"
            disabled={loading || !previewImage}
            className="w-full py-5 rounded-[2rem] font-black text-white text-xl shadow-2xl transition-all hover:scale-[1.02] active:scale-95 flex items-center justify-center gap-4"
            style={{
              background: `linear-gradient(135deg, ${accentColor}, #0ea5e9)`,
              boxShadow: `0 20px 40px -10px ${accentColor}60`,
              opacity: (loading || !previewImage) ? 0.6 : 1
            }}
          >
            {loading ? <FiLoader className="animate-spin" /> : <FiActivity />}
            {loading ? 'Checking...' : 'Check for Cancer'}
          </button>
        </form>
      </FormContainer>
    </DashboardLayout>
  );
};

export default BreastCancerPage;

import React, { useState } from 'react';
import { useForm, Controller } from 'react-hook-form';
import Select from 'react-select';
import { motion } from 'framer-motion';
import { useTheme } from '../context/ThemeContext';
import DashboardLayout from '../components/layout/DashboardLayout';
import FormContainer from '../components/ui/FormContainer';
import PredictionResult from '../components/ui/PredictionResult';
import {
  FiActivity,
  FiSearch,
  FiAlertCircle,
  FiCheckCircle,
  FiLoader,
  FiInfo,
  FiShield,
  FiZap
} from 'react-icons/fi';
import { predictDisease } from '../utils/api';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';

interface SymptomOption {
  value: string;
  label: string;
}

interface FormData {
  symptoms: SymptomOption[];
}

const SymptomPage: React.FC = () => {
  const { theme, accentColor } = useTheme();
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<any | null>(null);

  const schema = z.object({
    symptoms: z.array(z.object({ value: z.string(), label: z.string() }))
      .min(1, 'Please select at least one symptom')
  });

  const { control, handleSubmit, formState: { errors } } = useForm<FormData>({
    resolver: zodResolver(schema),
    defaultValues: { symptoms: [] }
  });

  const symptomOptions: SymptomOption[] = [
    { value: 'itching', label: 'Itching' },
    { value: 'skin_rash', label: 'Skin Rash' },
    { value: 'continuous_sneezing', label: 'Continuous Sneezing' },
    { value: 'shivering', label: 'Shivering' },
    { value: 'fatigue', label: 'Fatigue' },
    { value: 'cough', label: 'Cough' },
    { value: 'high_fever', label: 'High Fever' },
    { value: 'headache', label: 'Headache' },
    { value: 'nausea', label: 'Nausea' },
    { value: 'vomiting', label: 'Vomiting' },
    { value: 'muscle_pain', label: 'Muscle Pain' },
    { value: 'joint_pain', label: 'Joint Pain' },
  ];

  const onSubmit = async (data: FormData) => {
    setLoading(true);
    try {
      const symptoms = data.symptoms.map(option => option.value);
      const response = await predictDisease.symptom({ symptoms });

      if (response.error) {
        setResult({
          error: true,
          prediction: 'Analysis Failed',
          confidence: 0,
          description: 'Unable to reach the backend.',
          precautions: [],
          severity: 0
        });
        setLoading(false);
        return;
      }

      setResult({
        prediction: response.disease,
        confidence: response.confidence || 0.9,
        description: response.description || '',
        precautions: response.precautions || [],
        severity: response.severity?.score || 3
      });
    } catch (err) {
      setResult({
        error: true,
        prediction: 'Analysis Failed',
        confidence: 0,
        description: 'Unable to reach the backend.',
        precautions: [],
        severity: 0
      });
    } finally {
      setLoading(false);
    }
  };

  const selectStyles = {
    control: (provided: any, state: any) => ({
      ...provided,
      backgroundColor: theme === 'dark' ? 'rgba(255, 255, 255, 0.05)' : '#fff',
      borderColor: state.isFocused ? accentColor : (theme === 'dark' ? 'rgba(255,255,255,0.1)' : '#e2e8f0'),
      borderRadius: '1rem',
      padding: '0.5rem',
      boxShadow: state.isFocused ? `0 0 0 2px ${accentColor}20` : 'none',
      '&:hover': { borderColor: accentColor },
    }),
    multiValue: (base: any) => ({
      ...base,
      backgroundColor: `${accentColor}15`,
      borderRadius: '0.5rem',
      border: `1px solid ${accentColor}30`,
    }),
    multiValueLabel: (base: any) => ({
      ...base,
      color: accentColor,
      fontWeight: '700',
    })
  };

  return (
    <DashboardLayout>
      <FormContainer
        title="Symptom Checker"
        subtitle="Find out why you are feeling unwell"
        icon={<FiSearch size={28} />}
        predictionResult={result && (
          result.error ? (
            <PredictionResult
              prediction=""
              confidence={0}
              modelName="AI Engine"
              metrics={{ accuracy: 0, precision: 0, recall: 0, f1: 0 }}
              diseaseType="symptom"
              error={true}
            />
          ) : (
            <div className="space-y-8 h-full flex flex-col">
              <div className="text-center space-y-4">
                <div className="w-20 h-20 rounded-3xl bg-emerald-50 dark:bg-emerald-900/20 flex items-center justify-center mx-auto text-emerald-500 shadow-xl">
                  <FiCheckCircle size={40} />
                </div>
                <div>
                  <h4 className="text-sm font-black uppercase tracking-widest opacity-40">Possible Cause</h4>
                  <h2 className="text-4xl font-black tracking-tight" style={{ color: accentColor }}>{result.prediction}</h2>
                </div>
                <div className="flex justify-center">
                  <span className="px-6 py-2 rounded-full bg-slate-100 dark:bg-white/5 border border-slate-200 dark:border-white/10 text-xs font-black uppercase tracking-widest">
                    {Math.round(result.confidence * 100)}% Match
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 gap-4 overflow-y-auto max-h-[400px] pr-2 custom-scrollbar">
                <div className="p-6 rounded-3xl bg-slate-50 dark:bg-white/5 border border-slate-100 dark:border-white/5 space-y-3">
                  <div className="flex items-center gap-2 text-slate-800 dark:text-white font-bold">
                    <FiInfo /> What this means
                  </div>
                  <p className="text-sm leading-relaxed opacity-70">{result.description}</p>
                </div>

                <div className="p-6 rounded-3xl bg-emerald-50 dark:bg-emerald-900/10 border border-emerald-100 dark:border-emerald-500/20 space-y-4">
                  <div className="flex items-center gap-2 text-emerald-600 font-bold">
                    <FiShield /> What you should do
                  </div>
                  <div className="space-y-2">
                    {result.precautions.map((p: string, i: number) => (
                      <div key={i} className="flex items-center gap-3 text-sm font-medium">
                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                        {p}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="mt-auto pt-6 border-t border-slate-100 dark:border-white/5">
                <button onClick={() => setResult(null)} className="w-full py-4 rounded-2xl font-bold border-2 transition-all hover:bg-slate-50 dark:hover:bg-white/5 active:scale-95" style={{ borderColor: accentColor, color: accentColor }}>
                  Check Other Symptoms
                </button>
              </div>
            </div>
          )
        )}
      >
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-10 flex flex-col justify-center h-full pb-10">
          <div className="text-center space-y-4">
            <div className="w-16 h-16 rounded-2xl bg-slate-50 dark:bg-white/5 flex items-center justify-center mx-auto text-[#14b8a6] shadow-inner">
              <FiZap size={32} />
            </div>
            <h3 className="text-2xl font-black tracking-tight">How do you feel?</h3>
            <p className="text-sm opacity-50 font-medium max-w-sm mx-auto">Select the symptoms you have to find out what might be wrong.</p>
          </div>

          <div className="relative z-[100]">
            <Controller
              name="symptoms"
              control={control}
              render={({ field }) => (
                <Select
                  {...field}
                  isMulti
                  options={symptomOptions}
                  placeholder="Type symptoms here (e.g. fever, cough)..."
                  styles={selectStyles}
                  className="text-lg"
                />
              )}
            />
            {errors.symptoms && (
              <p className="text-red-500 text-xs mt-3 flex items-center justify-center gap-2 animate-pulse font-bold">
                <FiAlertCircle /> {errors.symptoms.message}
              </p>
            )}
          </div>

          <div className="pt-6">
            <button
              type="submit"
              disabled={loading}
              className="w-full py-5 rounded-[2rem] font-black text-white text-xl shadow-2xl transition-all hover:scale-[1.02] active:scale-95 flex items-center justify-center gap-4"
              style={{
                background: `linear-gradient(135deg, ${accentColor}, #0ea5e9)`,
                boxShadow: `0 20px 40px -10px ${accentColor}60`
              }}
            >
              {loading ? <FiLoader className="animate-spin" /> : <FiSearch />}
              {loading ? 'Checking...' : 'Check My Symptoms'}
            </button>
          </div>
        </form>
      </FormContainer>
    </DashboardLayout>
  );
};

export default SymptomPage;
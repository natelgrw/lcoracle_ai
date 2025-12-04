import React from 'react';
import { Link } from 'react-router-dom';
import { ChevronLeft } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

interface ModuleHeaderProps {
  title: string;
  description: string | React.ReactNode;
  icon: LucideIcon | string;
  color: string;
  gradient?: string;
  bg?: string;
  iconOffset?: string; // NEW PROP
}

const ModuleHeader: React.FC<ModuleHeaderProps> = ({
  title,
  description,
  icon: IconOrUrl,
  gradient,
  bg,
  iconOffset = ""
}) => {
  return (
    <div className={`w-full pt-24 pb-16 px-4 sm:px-6 lg:px-8 relative overflow-hidden ${gradient || bg || 'bg-slate-900'} text-white shadow-lg`}>
      {/* Decorative Background Pattern */}
      <div className="absolute inset-0 opacity-10 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] mix-blend-overlay"></div>

      <div className="max-w-7xl mx-auto relative z-10">
        <Link
          to="/"
          className="inline-flex items-center text-xs font-medium text-white/80 hover:text-white mb-6 transition-colors hover:-translate-x-1 duration-200"
        >
          <ChevronLeft className="w-3.5 h-3.5 mr-2" />
          Back to Dashboard
        </Link>

        <div className="flex flex-col md:flex-row items-start md:items-center gap-5">
          <div className={`rounded-xl bg-white/10 backdrop-blur-md border border-white/20 shadow-xl ${iconOffset} ${typeof IconOrUrl === 'string' ? 'p-0 overflow-hidden w-14 h-14' : 'p-3'}`}>
            {typeof IconOrUrl === 'string' ? (
              <img src={IconOrUrl} alt={title} className="w-full h-full object-cover" />
            ) : (
              <IconOrUrl className="w-8 h-8 text-white" />
            )}
          </div>

          <div>
            <h1 className="text-2xl md:text-4xl font-bold text-white mb-2 tracking-tight drop-shadow-sm">
              {title}
            </h1>
            <p className="text-base md:text-lg text-white/90 max-w-3xl font-light leading-relaxed drop-shadow-sm">
              {description}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModuleHeader;
import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, FlaskConical, Activity, Search, Layers, TrendingUp, Clock } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const Navbar: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navLinks = [
    { name: 'ASKCOS', path: '/askcos', icon: Search, color: 'text-blue-600' },
    { name: 'AMAX', path: '/amax', icon: Activity, color: 'text-purple-600' },
    { name: 'ReTiNA', path: '/retina', icon: Clock, color: 'text-blue-600' },
    { name: 'PeakProphet', path: '/peak-prophet', icon: Layers, color: 'text-purple-600' },
    { name: 'Gradience', path: '/gradience', icon: TrendingUp, color: 'text-blue-600' },
  ];

  return (
    <nav
      className={`fixed z-50 ease-in-out ${isOpen ? '' : 'transition-all duration-500'} ${scrolled && !isOpen
        ? 'top-4 left-4 right-4 rounded-full max-w-6xl mx-auto bg-white shadow-xl border border-gray-200/50 px-6 py-2'
        : 'top-0 left-0 right-0 bg-white shadow-sm px-4 sm:px-6 lg:px-8 py-3'
        } ${!scrolled && !isOpen ? 'bg-transparent shadow-none border-none' : ''}`}
    >
      <div className={`w-full ${!scrolled && 'max-w-7xl mx-auto'}`}>
        <div className="flex items-center justify-between h-10">

          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2 group">
            <div className="relative w-7 h-7 flex items-center justify-center bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg shadow-lg group-hover:scale-110 transition-transform">
              <FlaskConical className="w-4 h-4 text-white" />
            </div>
            <span className="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">
              LCOracle.ai
            </span>
          </Link>

          {/* Desktop Nav - Trigger hamburger on screens smaller than LG */}
          <div className="hidden lg:flex items-center space-x-6">
            {navLinks.map((link) => (
              <Link
                key={link.name}
                to={link.path}
                className={`flex items-center space-x-1 text-xs font-medium transition-colors hover:text-gray-900 ${location.pathname === link.path ? 'text-blue-600' : 'text-gray-600'
                  }`}
              >
                <link.icon className={`w-3.5 h-3.5 ${link.color}`} />
                <span>{link.name}</span>
              </Link>
            ))}
            <a
              href="#modules"
              className="px-4 py-1.5 rounded-full bg-blue-600 text-white text-xs font-bold shadow-md hover:bg-blue-700 hover:shadow-lg transition-all hover:scale-105"
            >
              Get Started
            </a>
          </div>

          {/* Mobile Menu Button */}
          <div className="lg:hidden">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="p-2 rounded-md text-gray-600 hover:text-gray-900 focus:outline-none"
            >
              {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0, marginTop: 0 }}
            animate={{ opacity: 1, height: 'auto', marginTop: 16 }}
            exit={{ opacity: 0, height: 0, marginTop: 0 }}
            className="lg:hidden bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden"
          >
            <div className="px-4 pt-2 pb-4 space-y-1">
              {navLinks.map((link) => (
                <Link
                  key={link.name}
                  to={link.path}
                  className="flex items-center space-x-3 px-3 py-3 rounded-xl text-base font-medium text-gray-700 hover:text-blue-600 hover:bg-blue-50 transition-colors"
                  onClick={() => setIsOpen(false)}
                >
                  <link.icon className={`w-5 h-5 ${link.color}`} />
                  <span>{link.name}</span>
                </Link>
              ))}
              <div className="pt-2">
                <a
                  href="#modules"
                  className="block w-full text-center px-4 py-3 rounded-xl bg-blue-600 text-white font-bold shadow-md"
                  onClick={() => setIsOpen(false)}
                >
                  Get Started
                </a>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
};

export default Navbar;

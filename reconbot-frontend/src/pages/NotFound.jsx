// src/pages/NotFound.jsx
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { useState, useEffect } from 'react';

const NotFound = () => {
  const [glitchActive, setGlitchActive] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    // Random glitch effect
    const interval = setInterval(() => {
      setGlitchActive(true);
      setTimeout(() => setGlitchActive(false), 200);
    }, Math.random() * 3000 + 2000);

    // Mouse tracking for interactive elements
    const handleMouseMove = (e) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };

    window.addEventListener('mousemove', handleMouseMove);

    return () => {
      clearInterval(interval);
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  const suggestions = [
    { text: 'Go to Dashboard', icon: '🏠', path: '/dashboard' },
    { text: 'View Privacy Policy', icon: '🔒', path: '/privacy-policy' },
    { text: 'Read Terms of Service', icon: '📋', path: '/terms-of-service' },
    { text: 'Contact Support', icon: '💬', href: 'mailto:support@matchledger.in' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center p-4 relative overflow-hidden">
      {/* Dynamic Background Elements */}
      <div className="absolute inset-0">
        {[...Array(50)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-red-400/30 rounded-full"
            animate={{
              x: [
                Math.random() * window.innerWidth,
                mousePosition.x + (Math.random() - 0.5) * 200,
                Math.random() * window.innerWidth,
              ],
              y: [
                Math.random() * window.innerHeight,
                mousePosition.y + (Math.random() - 0.5) * 200,
                Math.random() * window.innerHeight,
              ],
              opacity: [0.3, 0.8, 0.3],
              scale: [1, 1.5, 1],
            }}
            transition={{
              duration: Math.random() * 10 + 5,
              repeat: Infinity,
              repeatType: "reverse",
              ease: "easeInOut",
            }}
            initial={{
              x: Math.random() * window.innerWidth,
              y: Math.random() * window.innerHeight,
            }}
          />
        ))}
      </div>

      {/* Floating geometric shapes */}
      <div className="absolute inset-0">
        {[...Array(8)].map((_, i) => (
          <motion.div
            key={`shape-${i}`}
            className={`absolute ${
              i % 3 === 0 ? 'w-4 h-4 bg-purple-500/20 rounded-full' :
              i % 3 === 1 ? 'w-6 h-6 bg-pink-500/20 rotate-45' :
              'w-3 h-8 bg-blue-500/20 rounded-full'
            }`}
            animate={{
              x: [
                Math.random() * window.innerWidth,
                Math.random() * window.innerWidth,
              ],
              y: [
                Math.random() * window.innerHeight,
                Math.random() * window.innerHeight,
              ],
              rotate: [0, 360],
            }}
            transition={{
              duration: Math.random() * 20 + 10,
              repeat: Infinity,
              repeatType: "reverse",
            }}
            initial={{
              x: Math.random() * window.innerWidth,
              y: Math.random() * window.innerHeight,
            }}
          />
        ))}
      </div>

      <motion.div
        className="text-center max-w-4xl relative z-10"
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        {/* Animated 404 with Glitch Effect */}
        <motion.div
          className="relative mb-8"
          animate={{
            scale: glitchActive ? [1, 1.05, 0.98, 1.02, 1] : [1, 1.05, 1],
          }}
          transition={{
            duration: glitchActive ? 0.2 : 3,
            repeat: glitchActive ? 0 : Infinity,
            ease: "easeInOut"
          }}
        >
          <h1 className={`text-9xl md:text-[12rem] font-bold bg-gradient-to-r from-red-400 via-purple-400 to-pink-400 bg-clip-text text-transparent ${
            glitchActive ? 'animate-pulse' : ''
          }`}>
            404
          </h1>

          {/* Glitch overlay */}
          {glitchActive && (
            <motion.div
              className="absolute inset-0 text-9xl md:text-[12rem] font-bold text-red-500 opacity-70"
              style={{
                transform: 'translate(2px, -2px)',
                mixBlendMode: 'multiply',
              }}
              initial={{ opacity: 0 }}
              animate={{ opacity: [0, 0.7, 0] }}
              transition={{ duration: 0.2 }}
            >
              404
            </motion.div>
          )}
        </motion.div>

        {/* Error message with typewriter effect */}
        <motion.div
          className="mb-8"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Page Not Found
          </h2>
          <p className="text-xl md:text-2xl text-white/70 mb-2">
            The page you're looking for has vanished into the digital void.
          </p>
          <p className="text-lg text-white/50">
            Don't worry, even the best explorers sometimes take a wrong turn.
          </p>
        </motion.div>

        {/* Interactive search suggestions */}
        <motion.div
          className="mb-12"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
        >
          <h3 className="text-xl font-semibold text-white mb-6">
            Perhaps you were looking for one of these?
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {suggestions.map((suggestion, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1 + index * 0.1 }}
              >
                {suggestion.path ? (
                  <Link to={suggestion.path}>
                    <motion.div
                      className="bg-white/10 hover:bg-white/20 backdrop-blur-sm border border-white/20 rounded-xl p-4 transition-all duration-300 cursor-pointer group"
                      whileHover={{ scale: 1.05, y: -5 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <div className="text-3xl mb-2 group-hover:scale-110 transition-transform duration-300">
                        {suggestion.icon}
                      </div>
                      <p className="text-white font-medium">{suggestion.text}</p>
                    </motion.div>
                  </Link>
                ) : (
                  <motion.a
                    href={suggestion.href}
                    className="block"
                    whileHover={{ scale: 1.05, y: -5 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <div className="bg-white/10 hover:bg-white/20 backdrop-blur-sm border border-white/20 rounded-xl p-4 transition-all duration-300 cursor-pointer group">
                      <div className="text-3xl mb-2 group-hover:scale-110 transition-transform duration-300">
                        {suggestion.icon}
                      </div>
                      <p className="text-white font-medium">{suggestion.text}</p>
                    </div>
                  </motion.a>
                )}
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Main action buttons */}
        <motion.div
          className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5 }}
        >
          <Link to="/">
            <motion.button
              className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white px-8 py-4 rounded-xl font-medium transition-all duration-300 shadow-2xl group relative overflow-hidden"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <span className="relative z-10 flex items-center space-x-2">
                <svg className="w-5 h-5 group-hover:-translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"></path>
                </svg>
                <span>Go Home</span>
              </span>
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-pink-500 to-purple-500 opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                initial={false}
              />
            </motion.button>
          </Link>

          <motion.button
            onClick={() => window.history.back()}
            className="bg-white/10 hover:bg-white/20 text-white px-8 py-4 rounded-xl font-medium border border-white/20 transition-all duration-300 backdrop-blur-sm group"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <span className="flex items-center space-x-2">
              <svg className="w-5 h-5 group-hover:-translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7"></path>
              </svg>
              <span>Go Back</span>
            </span>
          </motion.button>
        </motion.div>

        {/* Fun error code display */}
        <motion.div
          className="bg-black/20 backdrop-blur-sm border border-red-400/20 rounded-xl p-6 max-w-2xl mx-auto"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.8 }}
        >
          <h4 className="text-red-300 font-mono text-sm mb-2">ERROR_DETAILS:</h4>
          <div className="text-left font-mono text-sm text-red-200/80 space-y-1">
            <div>STATUS: 404_NOT_FOUND</div>
            <div>TIMESTAMP: {new Date().toISOString()}</div>
            <div>URL: {window.location.pathname}</div>
            <div>USER_AGENT: {navigator.userAgent.split(' ')[0]}...</div>
            <div className="text-green-400">SUGGESTION: NAVIGATE_TO_VALID_ROUTE</div>
          </div>
        </motion.div>

        {/* Easter egg - Konami code hint */}
        <motion.div
          className="mt-8 text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2.2 }}
        >
          <p className="text-white/30 text-xs font-mono">
            💡 Pro tip: Try the Konami code for a surprise!
          </p>
        </motion.div>
      </motion.div>

      {/* Floating help button */}
      <motion.div
        className="fixed bottom-8 right-8 z-20"
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ delay: 2.5, type: "spring", stiffness: 200 }}
      >
        <motion.a
          href="mailto:support@matchledger.in"
          className="bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white p-4 rounded-full shadow-2xl backdrop-blur-sm border border-white/20"
          whileHover={{ scale: 1.1, rotate: 5 }}
          whileTap={{ scale: 0.9 }}
          title="Need help? Contact support"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
        </motion.a>
      </motion.div>
    </div>
  );
};

export default NotFound;
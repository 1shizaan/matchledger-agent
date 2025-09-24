// src/hooks/useSecureInput.js
import { useState, useCallback, useMemo } from 'react';

export const useSecureInput = (initialValue = '', options = {}) => {
  const [value, setValue] = useState(initialValue);
  const [error, setError] = useState(null);
  const [touched, setTouched] = useState(false);

  const {
    maxLength = 1000,
    minLength = 0,
    allowHtml = false,
    pattern = null,
    required = false,
    type = 'text', // text, email, password, number
    customValidator = null
  } = options;

  // Email validation regex
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  // Phone validation regex (basic international format)
  const phoneRegex = /^[\+]?[1-9][\d]{0,15}$/;

  const sanitizeInput = useCallback((input) => {
    if (typeof input !== 'string') return input;

    let sanitized = input;

    // Length validation
    if (sanitized.length > maxLength) {
      setError(`Input too long. Maximum ${maxLength} characters allowed.`);
      return value; // Keep previous value
    }

    if (required && sanitized.trim().length < minLength) {
      setError(`Input too short. Minimum ${minLength} characters required.`);
      return value;
    }

    // HTML sanitization (unless explicitly allowed)
    if (!allowHtml) {
      sanitized = sanitized
        .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
        .replace(/<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>/gi, '')
        .replace(/javascript:/gi, '')
        .replace(/on\w+\s*=/gi, '')
        .replace(/<[^>]*>/g, ''); // Remove all HTML tags
    }

    // Type-specific validation
    switch (type) {
      case 'email':
        if (sanitized && !emailRegex.test(sanitized)) {
          setError('Please enter a valid email address');
          return value;
        }
        break;

      case 'password':
        if (sanitized && sanitized.length > 0) {
          // Check for basic password strength
          const hasLower = /[a-z]/.test(sanitized);
          const hasUpper = /[A-Z]/.test(sanitized);
          const hasNumber = /\d/.test(sanitized);
          const hasSpecial = /[!@#$%^&*(),.?":{}|<>]/.test(sanitized);

          if (sanitized.length < 8) {
            setError('Password must be at least 8 characters long');
            return value;
          }

          if (!hasLower || !hasUpper || !hasNumber) {
            setError('Password must contain uppercase, lowercase, and numbers');
            return value;
          }
        }
        break;

      case 'number':
        if (sanitized && isNaN(sanitized)) {
          setError('Please enter a valid number');
          return value;
        }
        break;

      case 'phone':
        if (sanitized && !phoneRegex.test(sanitized.replace(/\s/g, ''))) {
          setError('Please enter a valid phone number');
          return value;
        }
        break;
    }

    // Pattern validation
    if (pattern && sanitized && !pattern.test(sanitized)) {
      setError('Invalid input format');
      return value;
    }

    // Custom validation
    if (customValidator && sanitized) {
      const customError = customValidator(sanitized);
      if (customError) {
        setError(customError);
        return value;
      }
    }

    // Required validation
    if (required && !sanitized.trim()) {
      setError('This field is required');
      return value;
    }

    setError(null);
    return sanitized;
  }, [value, maxLength, minLength, allowHtml, pattern, required, type, customValidator]);

  const handleChange = useCallback((newValue) => {
    const sanitized = sanitizeInput(newValue);
    setValue(sanitized);
    if (!touched) setTouched(true);
  }, [sanitizeInput, touched]);

  const handleBlur = useCallback(() => {
    setTouched(true);
    if (required && !value.trim()) {
      setError('This field is required');
    }
  }, [required, value]);

  const reset = useCallback(() => {
    setValue(initialValue);
    setError(null);
    setTouched(false);
  }, [initialValue]);

  const validate = useCallback(() => {
    const sanitized = sanitizeInput(value);
    return !error && sanitized === value;
  }, [value, sanitizeInput, error]);

  // Strength indicator for passwords
  const passwordStrength = useMemo(() => {
    if (type !== 'password' || !value) return null;

    let score = 0;
    const checks = {
      length: value.length >= 8,
      lowercase: /[a-z]/.test(value),
      uppercase: /[A-Z]/.test(value),
      numbers: /\d/.test(value),
      special: /[!@#$%^&*(),.?":{}|<>]/.test(value),
    };

    Object.values(checks).forEach(check => check && score++);

    const strength = score < 2 ? 'weak' : score < 4 ? 'medium' : 'strong';
    const percentage = (score / 5) * 100;

    return {
      score,
      strength,
      percentage,
      checks
    };
  }, [type, value]);

  return {
    value,
    error,
    touched,
    handleChange,
    handleBlur,
    reset,
    validate,
    isValid: !error && (!required || value.trim().length > 0),
    passwordStrength
  };
};

// Additional validation utilities
export const validators = {
  email: (value) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(value) ? null : 'Please enter a valid email address';
  },

  phone: (value) => {
    const phoneRegex = /^[\+]?[1-9][\d]{0,15}$/;
    return phoneRegex.test(value.replace(/\s/g, '')) ? null : 'Please enter a valid phone number';
  },

  url: (value) => {
    try {
      new URL(value);
      return null;
    } catch {
      return 'Please enter a valid URL';
    }
  },

  noSpecialChars: (value) => {
    const hasSpecialChars = /[<>:"/\\|?*]/.test(value);
    return hasSpecialChars ? 'Special characters are not allowed' : null;
  },

  alphanumeric: (value) => {
    const alphanumericRegex = /^[a-zA-Z0-9\s]*$/;
    return alphanumericRegex.test(value) ? null : 'Only letters, numbers, and spaces are allowed';
  },

  strongPassword: (value) => {
    const checks = {
      length: value.length >= 8,
      lowercase: /[a-z]/.test(value),
      uppercase: /[A-Z]/.test(value),
      numbers: /\d/.test(value),
      special: /[!@#$%^&*(),.?":{}|<>]/.test(value),
    };

    const passedChecks = Object.values(checks).filter(Boolean).length;

    if (passedChecks < 4) {
      return 'Password must contain uppercase, lowercase, numbers, and special characters';
    }

    return null;
  }
};

export default useSecureInput;
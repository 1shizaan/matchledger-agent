import * as Tooltip from '@radix-ui/react-tooltip';

const TooltipWrapper = ({ children, label }) => (
  <Tooltip.Provider>
    <Tooltip.Root>
      <Tooltip.Trigger asChild>
        {children}
      </Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content
          className="z-50 px-3 py-1.5 text-sm text-white bg-gray-900 rounded shadow-md"
          side="top"
          sideOffset={5}
        >
          {label}
          <Tooltip.Arrow className="fill-gray-900" />
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  </Tooltip.Provider>
);

export default TooltipWrapper;
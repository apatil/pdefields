! Subroutine EP is a template for a inference of a Gaussian Markov random field with known mean and precision via the EP algorithm. The template parameter LIKELIHOOD_CODE in braces should be replaced with a code snippet that evaluates the log-likelihood of a proposal, and the resulting code can be compiled either ahead of time or on the fly.

      subroutine ep (ind,dat,ptr,nnz,nx,lval,lvar,
     & diag, M, lv, nv, delta)
!f2py intent(hide) nnz, nx, nv
!f2py intent(out) lval, lvar, delta
      implicit none
      double precision dat(nnz), lval(nx), lvar(nx), M(nx)
      double precision diag(nx), mc(nx), lv(nx,nv), delta
      integer ind(nnz), ptr(nx+1), nx, nnz, i, j, nv


	  delta = 0.0D0
      do i=1,nx

!         mc is the conditional mean of x(i),
!         (Q(i,i) x(i) - Q(i,:) x) / Q(i,i)
          mc(i) = diag(i)*x(i)
          do j=ptr(i)+1,ptr(i+1)
              mc(i) = mc(i) - dat(j)*x(ind(j)+1)
          end do
          mc(i) = mc(i)/diag(i)
          
		  lval(i) = mc(i)
		  lvar(i) = mc(i)
          
      end do
      
      return
      end